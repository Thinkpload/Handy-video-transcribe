import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { commands } from "@/bindings";

type Segment = {
  start: number;
  end: number;
  speaker: string;
  text: string;
};

type Progress = {
  job_id: string;
  stage: string;
  processed_secs: number;
  total_secs: number;
};

function formatTime(secs: number): string {
  const s = Math.floor(secs % 60);
  const m = Math.floor((secs / 60) % 60);
  const h = Math.floor(secs / 3600);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function MeetingsSettings() {
  const { t } = useTranslation();
  const [ffmpegOk, setFfmpegOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    commands.checkFfmpegAvailable().then(setFfmpegOk).catch(() => setFfmpegOk(false));
  }, []);

  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    listen<Progress>("meeting-progress", (e) => {
      if (jobIdRef.current && e.payload.job_id !== jobIdRef.current) return;
      setProgress(e.payload);
    }).then((u) => (unlisten = u));
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const onPick = async () => {
    setError(null);
    const picked = await open({
      multiple: false,
      filters: [
        {
          name: t("meetings.fileFilter"),
          extensions: ["mp4", "mkv", "mov", "webm", "avi", "m4a", "mp3", "wav", "flac", "ogg"],
        },
      ],
    });
    if (!picked || typeof picked !== "string") return;
    const path = picked;
    setFileName(path.split(/[\\/]/).pop() ?? path);
    setSegments([]);
    setProgress(null);
    setBusy(true);
    const jobId = `meeting-${Date.now()}`;
    jobIdRef.current = jobId;
    try {
      const res = await commands.transcribeMeetingVideo(jobId, path);
      if (res.status === "error") {
        setError(res.error);
      } else {
        setSegments(res.data.segments);
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const exportText = () => {
    const body = segments
      .map((s) => `[${formatTime(s.start)}] ${s.speaker}: ${s.text}`)
      .join("\n");
    const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName ?? "transcript"}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pct =
    progress && progress.total_secs > 0
      ? Math.min(100, Math.round((progress.processed_secs / progress.total_secs) * 100))
      : 0;

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto">
      <div>
        <h2 className="text-xl font-semibold">{t("meetings.title")}</h2>
        <p className="text-sm opacity-70">{t("meetings.subtitle")}</p>
      </div>

      {ffmpegOk === false && (
        <div className="rounded border border-red-500/40 bg-red-500/10 p-3 text-sm">
          {t("meetings.ffmpegMissing")}
        </div>
      )}

      <div className="flex gap-2 items-center">
        <button
          onClick={onPick}
          disabled={busy || ffmpegOk === false}
          className="px-4 py-2 rounded bg-logo-primary text-white disabled:opacity-50"
        >
          {busy ? t("meetings.working") : t("meetings.pickFile")}
        </button>
        {segments.length > 0 && (
          <button
            onClick={exportText}
            className="px-4 py-2 rounded border border-mid-gray/40"
          >
            {t("meetings.exportTxt")}
          </button>
        )}
        {fileName && <span className="text-sm opacity-70 truncate">{fileName}</span>}
      </div>

      {busy && (
        <div className="flex flex-col gap-1">
          <div className="h-2 w-full bg-mid-gray/20 rounded overflow-hidden">
            <div
              className="h-full bg-logo-primary transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="text-xs opacity-70">
            {progress?.stage === "extract"
              ? t("meetings.stageExtract")
              : progress?.stage === "transcribe"
                ? `${t("meetings.stageTranscribe")} — ${formatTime(progress.processed_secs)} / ${formatTime(progress.total_secs)}`
                : t("meetings.working")}
          </div>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-500/40 bg-red-500/10 p-3 text-sm">
          {error}
        </div>
      )}

      {segments.length > 0 && (
        <div className="flex flex-col gap-3 mt-2">
          <div className="text-xs opacity-60">{t("meetings.diarizationNote")}</div>
          {segments.map((s, i) => (
            <div key={i} className="flex gap-3">
              <div className="w-16 shrink-0 text-xs font-mono opacity-60 pt-0.5">
                {formatTime(s.start)}
              </div>
              <div className="flex-1">
                <div className="text-xs font-semibold opacity-80">{s.speaker}</div>
                <div className="text-sm whitespace-pre-wrap">{s.text}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
