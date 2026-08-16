import { afterEach, describe, expect, it } from "bun:test";
import * as fsp from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { BlobStore, resolveImageDataSync } from "@oh-my-pi/pi-coding-agent/session/blob-store";
import { buildSessionContext } from "@oh-my-pi/pi-coding-agent/session/session-context";
import type { SessionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { loadEntriesFromFile } from "@oh-my-pi/pi-coding-agent/session/session-loader";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { getBlobsDir } from "@oh-my-pi/pi-utils";
import * as snapcompact from "@oh-my-pi/snapcompact";

/** Under the 500k persistence truncation limit, over the 1 KiB blob threshold. */
const FRAME_DATA = "Zm".repeat(60_000);
const FRAMES_PER_ARCHIVE = 4;

function makeAssistantMessage(text: string) {
	const model = getBundledModel("anthropic", "claude-sonnet-4-5");
	if (!model) throw new Error("Expected built-in anthropic model to exist");
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop" as const,
		timestamp: 2,
	};
}

function makeArchive(marker: string) {
	return {
		[snapcompact.PRESERVE_KEY]: {
			frames: Array.from({ length: FRAMES_PER_ARCHIVE }, () => ({
				data: FRAME_DATA,
				mimeType: "image/png",
				cols: 196,
				rows: 71,
				chars: 13_916,
				font: "8x13",
				variant: "bw",
			})),
			totalChars: 13_916,
			truncatedChars: 0,
			text: `${marker}-source`,
			textHead: `${marker}-head`,
			textTail: `${marker}-tail`,
		},
		openaiRemoteCompaction: {
			provider: "openai",
			replacementHistory: [
				{ type: "message", role: "user", content: [{ type: "input_text", text: `preserved ${marker}` }] },
			],
		},
	};
}

function archiveFrames(session: SessionManager, entryId: string): snapcompact.Frame[] {
	const entry = session.getEntry(entryId);
	if (entry?.type !== "compaction") throw new Error(`Expected compaction ${entryId}`);
	const archive = snapcompact.getPreservedArchive(entry.preserveData);
	if (!archive) throw new Error(`Expected an archive on ${entryId}`);
	return archive.frames;
}

function imageDataIn(messages: readonly unknown[]): string[] {
	const found: string[] = [];
	const walk = (value: unknown): void => {
		if (Array.isArray(value)) {
			for (const item of value) walk(item);
			return;
		}
		if (!value || typeof value !== "object") return;
		if ("type" in value && value.type === "image" && "data" in value && typeof value.data === "string") {
			found.push(value.data);
			return;
		}
		for (const item of Object.values(value)) walk(item);
	};
	walk(messages);
	return found;
}

describe("snapcompact frames in the blob store", () => {
	const tempDirs: string[] = [];

	async function makeTempDir(): Promise<string> {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "omp-frame-blobs-"));
		tempDirs.push(dir);
		return dir;
	}

	async function writeJournal(): Promise<{
		sessionFile: string;
		sessionDir: string;
		firstId: string;
		secondId: string;
	}> {
		const dir = await makeTempDir();
		const sessionDir = path.join(dir, "sessions");
		const session = SessionManager.create(dir, sessionDir);
		const anchor = session.appendMessage({ role: "user", content: "hello", timestamp: 1 });
		session.appendMessage(makeAssistantMessage("hi"));
		const firstId = session.appendCompaction(
			"first",
			undefined,
			anchor,
			1000,
			undefined,
			undefined,
			makeArchive("first"),
		);
		const between = session.appendMessage({ role: "user", content: "between", timestamp: 3 });
		const secondId = session.appendCompaction(
			"second",
			undefined,
			between,
			2000,
			undefined,
			undefined,
			makeArchive("second"),
		);
		await session.flush();
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a persisted session file");
		await session.close();
		return { sessionFile, sessionDir, firstId, secondId };
	}

	function reopen(sessionFile: string, sessionDir: string): Promise<SessionManager> {
		return SessionManager.open(sessionFile, sessionDir, undefined, { suppressBreadcrumb: true });
	}

	afterEach(async () => {
		await Promise.all(tempDirs.map(dir => fsp.rm(dir, { recursive: true, force: true })));
		tempDirs.length = 0;
	});

	it("persists frames as blob references instead of inline base64", async () => {
		const { sessionFile } = await writeJournal();
		const body = await Bun.file(sessionFile).text();

		expect(body).not.toContain(FRAME_DATA);
		expect(body).toContain("blob:sha256:");
		// Two archives of identical frames collapse onto one content-addressed blob.
		const refs = new Set(body.match(/blob:sha256:[0-9a-f]{64}/g) ?? []);
		expect(refs.size).toBe(1);
		expect(body.length).toBeLessThan(FRAME_DATA.length);
	});

	it("keeps frame payloads out of memory after a resume", async () => {
		const { sessionFile, sessionDir, firstId, secondId } = await writeJournal();
		const reopened = await reopen(sessionFile, sessionDir);

		for (const id of [firstId, secondId]) {
			for (const frame of archiveFrames(reopened, id)) {
				expect(frame.data.startsWith("blob:sha256:")).toBe(true);
			}
		}
		// Everything else in preserveData is untouched by this change.
		const entry = reopened.getEntry(firstId);
		if (entry?.type !== "compaction") throw new Error("Expected compaction");
		expect(entry.preserveData?.openaiRemoteCompaction).toEqual(makeArchive("first").openaiRemoteCompaction);

		await reopened.close();
	});

	it("builds the same context as inline frames, before and after a rewind", async () => {
		const { sessionFile, sessionDir, firstId } = await writeJournal();
		const reopened = await reopen(sessionFile, sessionDir);
		const blobs = new BlobStore(getBlobsDir());

		/** Control: the same entries with every frame ref resolved back inline. */
		const inlineContext = (leafId: string | undefined) => {
			const entries = reopened.getEntries().map(entry => {
				if (entry.type !== "compaction") return entry;
				const archive = snapcompact.getPreservedArchive(entry.preserveData);
				if (!archive) return entry;
				return {
					...entry,
					preserveData: {
						...entry.preserveData,
						[snapcompact.PRESERVE_KEY]: {
							...archive,
							frames: archive.frames.map(frame => ({ ...frame, data: resolveImageDataSync(blobs, frame.data) })),
						},
					},
				} satisfies SessionEntry;
			});
			return buildSessionContext(entries, leafId, undefined, {}).messages;
		};

		const newestLeaf = reopened.getLeafId() ?? undefined;
		expect(imageDataIn(inlineContext(newestLeaf))).toContain(FRAME_DATA);
		expect(imageDataIn(reopened.buildSessionContext().messages)).toEqual(imageDataIn(inlineContext(newestLeaf)));

		reopened.branch(firstId);
		expect(imageDataIn(inlineContext(firstId))).toContain(FRAME_DATA);
		expect(imageDataIn(reopened.buildSessionContext().messages)).toEqual(imageDataIn(inlineContext(firstId)));

		await reopened.close();
	});

	it("drops a frame whose blob went missing instead of sending the reference", async () => {
		const { sessionFile, sessionDir } = await writeJournal();
		const body = await Bun.file(sessionFile).text();
		const ref = body.match(/blob:sha256:([0-9a-f]{64})/);
		if (!ref) throw new Error("Expected a frame blob reference");
		await fsp.rm(path.join(getBlobsDir(), ref[1]!), { force: true });

		const reopened = await reopen(sessionFile, sessionDir);
		const images = imageDataIn(reopened.buildSessionContext().messages);
		expect(images.some(data => data.startsWith("blob:sha256:"))).toBe(false);
		expect(images).not.toContain(FRAME_DATA);

		await reopened.close();
	});

	it("still reads a legacy journal that stored frames inline", async () => {
		const { sessionFile, firstId } = await writeJournal();
		const legacyDir = await makeTempDir();
		const legacySessionDir = path.join(legacyDir, "sessions");
		await fsp.mkdir(legacySessionDir, { recursive: true });
		const legacyFile = path.join(legacySessionDir, path.basename(sessionFile));
		const blobs = new BlobStore(getBlobsDir());
		// Rewrite the journal the way pre-change OMP wrote it: frames inline.
		const legacyBody = (await Bun.file(sessionFile).text()).replaceAll(/blob:sha256:[0-9a-f]{64}/g, match =>
			resolveImageDataSync(blobs, match),
		);
		await Bun.write(legacyFile, legacyBody);

		const reopened = await reopen(legacyFile, legacySessionDir);
		expect(archiveFrames(reopened, firstId)[0]?.data).toBe(FRAME_DATA);
		expect(imageDataIn(reopened.buildSessionContext().messages)).toContain(FRAME_DATA);

		await reopened.close();
		// A rewrite of that legacy journal externalizes the frames it had inline.
		const entries = await loadEntriesFromFile(legacyFile);
		expect(entries.length).toBeGreaterThan(0);
	});
});
