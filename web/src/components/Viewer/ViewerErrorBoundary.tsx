"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Bump this value (e.g. pass a URL) to reset the error state when the
   *  underlying asset changes. */
  resetKey?: string | null;
}

interface State {
  error: Error | null;
}

/**
 * Catches exceptions anywhere in the viewer subtree (Canvas, loaders, mesh
 * layers) so a malformed GLB / PLY / WebGL-context loss doesn't nuke the
 * whole page with Next.js's "Something went wrong — reload" overlay.
 */
export class ViewerErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: unknown) {
    // Log for dev visibility but don't rethrow.
    console.error("viewer crash:", error, info);
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            padding: 20,
            background: "var(--bg)",
            color: "var(--fg)",
            textAlign: "center",
            gap: 12,
          }}
        >
          <div
            style={{
              fontSize: "var(--fs-xs)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--danger)",
            }}
          >
            viewer crashed
          </div>
          <div
            style={{
              fontSize: "var(--fs-xs)",
              maxWidth: 520,
              color: "var(--muted)",
              whiteSpace: "pre-wrap",
            }}
          >
            {this.state.error.message || String(this.state.error)}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={() => this.setState({ error: null })}
            >
              retry
            </button>
            <button type="button" onClick={() => window.location.reload()}>
              reload page
            </button>
          </div>
          <div className="mono-small" style={{ color: "var(--muted)", maxWidth: 480 }}>
            If this keeps happening, download the GLB/PLY from the export
            panel and open it in Blender or MeshLab — the file itself is
            probably fine, the browser just can&rsquo;t render it.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
