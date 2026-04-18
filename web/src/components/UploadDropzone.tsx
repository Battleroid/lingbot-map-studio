"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  files: File[];
  onChange: (files: File[]) => void;
  disabled?: boolean;
}

const VIDEO_EXT = /\.(mp4|mov|mkv|webm|avi)$/i;

export function UploadDropzone({ files, onChange, disabled }: Props) {
  const [active, setActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback(
    (incoming: FileList | File[]) => {
      const list = Array.from(incoming).filter(
        (f) => f.type.startsWith("video/") || VIDEO_EXT.test(f.name),
      );
      if (!list.length) return;
      onChange([...files, ...list]);
    },
    [files, onChange],
  );

  return (
    <div>
      <div
        className="dropzone"
        data-active={active && !disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setActive(true);
        }}
        onDragLeave={() => setActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setActive(false);
          if (disabled) return;
          addFiles(e.dataTransfer.files);
        }}
      >
        <div className="primary">drop video(s)</div>
        <div className="secondary">mp4 · mov · mkv · webm — order = scene order</div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/*,.mp4,.mov,.mkv,.webm,.avi"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {files.length > 0 && (
        <table className="grid" style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th style={{ width: 24 }}>#</th>
              <th>file</th>
              <th style={{ width: 100 }}>size</th>
              <th style={{ width: 30 }}></th>
            </tr>
          </thead>
          <tbody>
            {files.map((f, i) => (
              <tr key={`${f.name}-${i}`}>
                <td>{i + 1}</td>
                <td style={{ wordBreak: "break-all" }}>{f.name}</td>
                <td>{(f.size / 1024 / 1024).toFixed(1)} MB</td>
                <td>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => onChange(files.filter((_, j) => j !== i))}
                  >
                    x
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
