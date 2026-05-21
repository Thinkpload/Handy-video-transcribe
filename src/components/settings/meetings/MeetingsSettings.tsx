import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { commands, type MeetingSummary } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { exportSegments, type ExportFormat } from "./exporters";

const LANGUAGES: { code: string; label: string }[] = [
  { code: "auto", label: "Auto-detect" },
  { code: "en", label: "English" },
  { code: "ru", label: "Русский" },
  { code: "es", label: "Español" },
  { code: "de", label: "Deutsch" },
  { code: "fr", label: "Français" },
  { code: "uk", label: "Українська" },
  { code: "zh", label: "中文" },
  { code: "ja", label: "日本語" },
];

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
  const { settings } = useSettings();
  const currentLang = settings?.selected_language ?? "auto";
  const [ffmpegOk, setFfmpegOk] = useState<boolean | null>(null);
  const [diarReady, setDiarReady] = useState<boolean | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [numSpeakers, setNumSpeakers] = useState<number>(0);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [savedMeetings, setSavedMeetings] = useState<MeetingSummary[]>([]);
  const jobIdRef = useRef<string | null>(null);

  const refreshSaved = useCallback(async () => {
    const res = await commands.listMeetings();
    if (res.status === "ok") setSavedMeetings(res.data);
  }, []);

  useEffect(() => {
    refreshSaved();
  }, [refreshSaved]);

  const openSaved = async (id: number) => {
    const res = await commands.getMeeting(id);
    if (res.status === "ok" && res.data) {
      setSegments(res.data.segments);
      setFileName(res.data.file_name);
      setError(null);
    }
  };

  const deleteSaved = async (id: number) => {
    await commands.deleteMeeting(id);
    refreshSaved();
  };

  useEffect(() => {
    commands.checkFfmpegAvailable().then(setFfmpegOk).catch(() => setFfmpegOk(false));
    commands.diarizationModelsPresent().then(setDiarReady).catch(() => setDiarReady(false));
  }, []);

  const onDownloadModels = async () => {
    setDownloading(true);
    setError(null);
    try {
      const res = await commands.downloadDiarizationModels();
      if (res.status === "error") setError(res.error);
      else setDiarReady(true);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setDownloading(false);
    }
  };

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
      const res = await commands.transcribeMeetingVideo(jobId, path, numSpeakers || null);
      if (res.status === "error") {
        setError(res.error);
      } else {
        setSegments(res.data.segments);
        refreshSaved();
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const doExport = (fmt: ExportFormat) => {
    exportSegments(segments, fileName ?? "transcript", fmt);
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

      <div className="rounded border border-mid-gray/30 p-3 text-sm flex items-center justify-between gap-3">
        <div>
          <div className="font-medium">{t("meetings.diarizationModels")}</div>
          <div className="opacity-70 text-xs">
            {diarReady
              ? t("meetings.diarizationReady")
              : t("meetings.diarizationMissing")}
          </div>
        </div>
        {!diarReady && (
          <button
            onClick={onDownloadModels}
            disabled={downloading}
            className="px-3 py-1.5 rounded border border-mid-gray/40 disabled:opacity-50"
          >
            {downloading ? t("meetings.downloading") : t("meetings.downloadModels")}
          </button>
        )}
      </div>

      <div className="flex gap-2 items-center flex-wrap">
        <label className="text-sm flex items-center gap-2">
          {t("meetings.language")}
          <select
            value={currentLang}
            disabled={busy}
            onChange={(e) => commands.changeSelectedLanguageSetting(e.target.value)}
            className="px-2 py-1 rounded border border-mid-gray/40 bg-transparent text-sm"
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm flex items-center gap-2">
          {t("meetings.numSpeakers")}
          <input
            type="number"
            min={0}
            max={20}
            value={numSpeakers}
            disabled={busy}
            onChange={(e) => setNumSpeakers(Number(e.target.value) || 0)}
            className="w-16 px-2 py-1 rounded border border-mid-gray/40 bg-transparent text-sm"
            title={t("meetings.numSpeakersHint")}
          />
        </label>
        <button
          onClick={onPick}
          disabled={busy || ffmpegOk === false}
          className="px-4 py-2 rounded bg-logo-primary text-white disabled:opacity-50"
        >
          {busy ? t("meetings.working") : t("meetings.pickFile")}
        </button>
        {segments.length > 0 && (
          <div className="flex gap-1">
            {(["txt", "srt", "vtt", "md", "json"] as ExportFormat[]).map((f) => (
              <button
                key={f}
                onClick={() => doExport(f)}
                className="px-2 py-1.5 rounded border border-mid-gray/40 text-xs uppercase"
              >
                {f}
              </button>
            ))}
          </div>
        )}
        {fileName && <span className="text-sm opacity-70 truncate">{fileName}</span>}
      </div>

      {savedMeetings.length > 0 && (
        <div className="rounded border border-mid-gray/30">
          <div className="px-3 py-2 text-xs font-semibold opacity-70 border-b border-mid-gray/20">
            {t("meetings.savedHeading")}
          </div>
          <div className="divide-y divide-mid-gray/20 max-h-48 overflow-y-auto">
            {savedMeetings.map((m) => (
              <div key={m.id} className="flex items-center gap-2 px-3 py-2 text-sm">
                <button
                  onClick={() => openSaved(m.id)}
                  className="flex-1 text-left hover:underline truncate"
                  title={m.source_path}
                >
                  <span className="font-medium">{m.file_name}</span>
                  <span className="opacity-60 ml-2 text-xs">
                    {new Date(m.created_at * 1000).toLocaleString()} ·{" "}
                    {formatTime(m.duration_secs)} · {m.speaker_count}{" "}
                    {t("meetings.speakersShort")}
                  </span>
                </button>
                <button
                  onClick={() => deleteSaved(m.id)}
                  className="px-2 py-1 text-xs opacity-60 hover:opacity-100"
                  title={t("meetings.delete")}
                >
                  {t("meetings.deleteIcon")}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

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
