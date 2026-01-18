import { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';
import {
  loadNiftiWithCache,
  extractSlice,
  getSliceDims,
  getMaxSliceIndex,
  type NiftiVolume,
} from '../lib/nifti';
import {
  loadSessionMetadata,
  getSessionForWeek,
  type SessionMetadata,
} from '../lib/meshData';

type ViewType = 'sagittal' | 'coronal' | 'axial';

interface MriViewerProps {
  week: number;
}

export default function MriViewer({ week }: MriViewerProps) {
  const [volume, setVolume] = useState<NiftiVolume | null>(null);
  const [metadata, setMetadata] = useState<SessionMetadata | null>(null);
  const [view, setView] = useState<ViewType>('sagittal');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSession, setCurrentSession] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const meta = await loadSessionMetadata();
        setMetadata(meta);
      } catch (err) {
        console.error('Failed to load metadata:', err);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!metadata) return;

    const sessionId = getSessionForWeek(metadata, week);
    if (!sessionId || sessionId === currentSession) return;

    const url = metadata.sessionUrls[sessionId];
    if (!url) {
      setError('No MRI data available for this week');
      return;
    }

    async function loadVolume() {
      setLoading(true);
      setError(null);

      try {
        const vol = await loadNiftiWithCache(sessionId!, url);
        setVolume(vol);
        setCurrentSession(sessionId);

        const maxSlice = getMaxSliceIndex(vol.dims, view);
        setSliceIndex(Math.floor(maxSlice / 2));
      } catch (err) {
        console.error('Failed to load MRI volume:', err);
        setError('Failed to load MRI data');
      } finally {
        setLoading(false);
      }
    }

    loadVolume();
  }, [week, metadata, currentSession, view]);

  useEffect(() => {
    if (!volume) return;
    const maxSlice = getMaxSliceIndex(volume.dims, view);
    setSliceIndex(Math.floor(maxSlice / 2));
  }, [view, volume]);

  const getSliceData = useCallback((): number[][] | null => {
    if (!volume) return null;

    const slice = extractSlice(volume, view, sliceIndex);
    const [width, height] = getSliceDims(volume.dims, view);

    const data: number[][] = [];
    for (let y = 0; y < height; y++) {
      const row: number[] = [];
      for (let x = 0; x < width; x++) {
        row.push(slice[y * width + x]);
      }
      data.push(row);
    }
    return data;
  }, [volume, view, sliceIndex]);

  const plotData: Data[] = volume
    ? [
        {
          type: 'heatmap',
          z: getSliceData() || [],
          colorscale: 'Greys',
          showscale: false,
          hoverinfo: 'skip',
        } as Data,
      ]
    : [];

  const sliceDims = volume ? getSliceDims(volume.dims, view) : [100, 100];
  const aspectRatio = sliceDims[1] / sliceDims[0];
  const baseWidth = 260;
  const layout: Partial<Layout> = {
    width: baseWidth,
    height: baseWidth * aspectRatio,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    xaxis: {
      visible: false,
      showgrid: false,
      zeroline: false,
      showticklabels: false,
    },
    yaxis: {
      visible: false,
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      scaleanchor: 'x',
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  };

  const maxSlice = volume ? getMaxSliceIndex(volume.dims, view) : 100;

  return (
    <div className="viz-card flex flex-col p-4">
      <div className="section-label mb-2">MRI Scan</div>
      
      {/* MRI Display - Compact */}
      <div className="flex-1 flex items-center justify-center bg-herbrain-dark rounded-xl overflow-hidden relative" style={{ minHeight: '200px' }}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-herbrain-dark/90 z-10">
            <div className="loading-spinner"></div>
          </div>
        )}

        {error ? (
          <div className="flex flex-col items-center justify-center p-4 text-center">
            <svg className="w-6 h-6 text-herbrain-muted/40 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-herbrain-muted/60 text-[10px]">{error}</p>
          </div>
        ) : volume ? (
          <Plot
            data={plotData}
            layout={layout}
            config={{
              displayModeBar: false,
              staticPlot: true,
            }}
          />
        ) : (
          <div className="flex items-center justify-center">
            <p className="text-herbrain-muted/40 text-[10px]">Loading MRI...</p>
          </div>
        )}
      </div>

      {/* View Selection - Pill Style */}
      <div className="flex items-center justify-center gap-1 mt-3 mb-2">
        {(['sagittal', 'coronal', 'axial'] as ViewType[]).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`px-3 py-1.5 text-[11px] font-medium rounded-lg transition-all duration-150 ${
              view === v
                ? 'bg-herbrain-green text-white shadow-sm'
                : 'text-herbrain-muted hover:bg-herbrain-surface'
            }`}
          >
            {v.charAt(0).toUpperCase() + v.slice(1)}
          </button>
        ))}
      </div>

      {/* Slice Slider - Minimal */}
      <div className="px-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-[9px] text-herbrain-muted/60 uppercase tracking-wide">Slice</span>
          <span className="text-[10px] text-herbrain-muted tabular-nums">
            {sliceIndex} / {maxSlice}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={maxSlice}
          value={sliceIndex}
          onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
          className="premium-slider w-full"
          disabled={!volume || loading}
        />
      </div>
    </div>
  );
}
