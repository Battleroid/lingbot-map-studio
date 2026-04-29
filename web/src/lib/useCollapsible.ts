"use client";

import { useState, type CSSProperties, type KeyboardEvent } from "react";

/**
 * Tiny shared hook for the side-panel collapsibility used on the job
 * page. Each panel renders its own `<div className="panel"><div
 * className="panel-header">…</div><div className="panel-body">…</div></div>`
 * shape; this hook returns prop bags to spread onto those two elements
 * so a single click on the header toggles the body's visibility.
 *
 * The CSS that actually hides the body is in `globals.css`:
 *   .panel[data-collapsed="true"] .panel-body { display: none; }
 *
 * Usage inside a component that takes a `collapsible?: boolean` prop:
 *
 *   const c = useCollapsible({ enabled: collapsible, initial: false });
 *   return (
 *     <div className="panel" {...c.panelProps}>
 *       <div className="panel-header" {...c.headerProps}>
 *         {c.arrow}<span>title</span>
 *       </div>
 *       <div className="panel-body">…</div>
 *     </div>
 *   );
 *
 * When `enabled` is false (default), the prop bags are inert and the
 * component renders exactly as it did before — the home page (and any
 * other non-collapsible context) doesn't pay any cost.
 */
export interface CollapsibleHandles {
  collapsed: boolean;
  panelProps: { "data-collapsed"?: "true" | "false" };
  headerProps: {
    "data-collapsible"?: "true";
    onClick?: () => void;
    onKeyDown?: (e: KeyboardEvent<HTMLElement>) => void;
    role?: "button";
    tabIndex?: 0;
    style?: CSSProperties;
  };
  /** Inline arrow glyph to render inside the header. Empty string when
   *  collapsibility is off so callers can just `{c.arrow}` unconditionally. */
  arrow: string;
}

export function useCollapsible(opts?: {
  enabled?: boolean;
  initial?: boolean;
}): CollapsibleHandles {
  const enabled = opts?.enabled ?? true;
  const [collapsed, setCollapsed] = useState(opts?.initial ?? false);

  if (!enabled) {
    return { collapsed: false, panelProps: {}, headerProps: {}, arrow: "" };
  }

  const toggle = () => setCollapsed((v) => !v);
  return {
    collapsed,
    panelProps: { "data-collapsed": collapsed ? "true" : "false" },
    headerProps: {
      "data-collapsible": "true",
      onClick: toggle,
      onKeyDown: (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      },
      role: "button",
      tabIndex: 0,
    },
    arrow: collapsed ? "▸ " : "▾ ",
  };
}
