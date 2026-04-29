import type { JobEvent } from "./types";

type Listener = (ev: JobEvent) => void;
type StatusListener = (status: "connecting" | "open" | "closed") => void;

export class JobStreamClient {
  private url: string;
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private statusListeners = new Set<StatusListener>();
  private closed = false;
  private reconnectMs = 1000;

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.closed) return;
    this.notifyStatus("connecting");
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.addEventListener("open", () => {
      this.reconnectMs = 1000;
      this.notifyStatus("open");
    });
    ws.addEventListener("message", (e) => {
      try {
        const ev = JSON.parse(e.data as string) as JobEvent;
        for (const l of this.listeners) l(ev);
      } catch (err) {
        console.warn("bad ws payload", err);
      }
    });
    ws.addEventListener("close", (e) => {
      this.notifyStatus("closed");
      if (this.closed) return;
      // Server closes the WS cleanly (code 1000) once the job's events.jsonl
      // has been replayed AND the events.done sentinel exists — i.e. the job
      // is in a terminal state and there will never be more events. Treat
      // that as "stop reconnecting" so the jobs page doesn't loop
      // connecting → closed → connecting → closed forever after a job
      // finishes. Unclean closes (network blip, server crash, etc.) keep
      // the existing exponential reconnect.
      if (e.code === 1000) {
        this.closed = true;
        return;
      }
      const delay = this.reconnectMs;
      this.reconnectMs = Math.min(10_000, this.reconnectMs * 2);
      setTimeout(() => this.connect(), delay);
    });
    ws.addEventListener("error", () => {
      try {
        ws.close();
      } catch {
        /* noop */
      }
    });
  }

  onEvent(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  onStatus(l: StatusListener): () => void {
    this.statusListeners.add(l);
    return () => this.statusListeners.delete(l);
  }

  private notifyStatus(status: "connecting" | "open" | "closed") {
    for (const l of this.statusListeners) l(status);
  }

  close(): void {
    this.closed = true;
    if (this.ws) this.ws.close();
  }
}
