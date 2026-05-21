import type { MeetingSegment } from "@/bindings";

export type ExportFormat = "txt" | "srt" | "vtt" | "md" | "json";

function pad(n: number, w = 2): string {
  return n.toString().padStart(w, "0");
}

function srtTimestamp(secs: number): string {
  const ms = Math.round(secs * 1000);
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  const millis = ms % 1000;
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(millis, 3)}`;
}

function vttTimestamp(secs: number): string {
  return srtTimestamp(secs).replace(",", ".");
}

function clockTimestamp(secs: number): string {
  const s = Math.floor(secs % 60);
  const m = Math.floor((secs / 60) % 60);
  const h = Math.floor(secs / 3600);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function toTxt(segments: MeetingSegment[]): string {
  return segments
    .map((s) => `[${clockTimestamp(s.start)}] ${s.speaker}: ${s.text}`)
    .join("\n");
}

export function toSrt(segments: MeetingSegment[]): string {
  return segments
    .map(
      (s, i) =>
        `${i + 1}\n${srtTimestamp(s.start)} --> ${srtTimestamp(s.end)}\n${s.speaker}: ${s.text}\n`,
    )
    .join("\n");
}

export function toVtt(segments: MeetingSegment[]): string {
  const body = segments
    .map(
      (s) =>
        `${vttTimestamp(s.start)} --> ${vttTimestamp(s.end)}\n<v ${s.speaker}>${s.text}\n`,
    )
    .join("\n");
  return `WEBVTT\n\n${body}`;
}

export function toMarkdown(segments: MeetingSegment[], title?: string): string {
  const header = title ? `# ${title}\n\n` : "";
  const grouped: { speaker: string; start: number; end: number; lines: string[] }[] = [];
  for (const s of segments) {
    const last = grouped[grouped.length - 1];
    if (last && last.speaker === s.speaker) {
      last.lines.push(s.text);
      last.end = s.end;
    } else {
      grouped.push({ speaker: s.speaker, start: s.start, end: s.end, lines: [s.text] });
    }
  }
  const body = grouped
    .map(
      (g) =>
        `### ${g.speaker}  _[${clockTimestamp(g.start)}]_\n\n${g.lines.join(" ")}\n`,
    )
    .join("\n");
  return header + body;
}

export function toJson(segments: MeetingSegment[]): string {
  return JSON.stringify(segments, null, 2);
}

export function downloadBlob(
  content: string,
  filename: string,
  mime = "text/plain;charset=utf-8",
) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportSegments(
  segments: MeetingSegment[],
  fileName: string,
  fmt: ExportFormat,
) {
  const base = fileName.replace(/\.[^.]+$/, "") || "transcript";
  switch (fmt) {
    case "txt":
      return downloadBlob(toTxt(segments), `${base}.txt`);
    case "srt":
      return downloadBlob(toSrt(segments), `${base}.srt`);
    case "vtt":
      return downloadBlob(toVtt(segments), `${base}.vtt`, "text/vtt;charset=utf-8");
    case "md":
      return downloadBlob(toMarkdown(segments, base), `${base}.md`, "text/markdown;charset=utf-8");
    case "json":
      return downloadBlob(toJson(segments), `${base}.json`, "application/json;charset=utf-8");
  }
}
