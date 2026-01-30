import { useState, useEffect, useCallback, useRef } from 'react';
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
  const [containerWidth, setContainerWidth] = useState(260);
  const containerRef = useRef<HTMLDivElement>(null);

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
      setError('No MRI data available');
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
        setError('Failed to load MRI');
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

  // Responsive sizing
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateWidth = () => {
      const rect = container.getBoundingClientRect();
      const newWidth = Math.max(180, Math.min(rect.width - 24, 320));
      setContainerWidth(newWidth);
    };

    updateWidth();
    
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(container);
    
    return () => resizeObserver.disconnect();
  }, []);

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
  const layout: Partial<Layout> = {
    width: containerWidth,
    height: containerWidth * aspectRatio,
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
    <div className="viz-card flex flex-col h-full p-4 sm:p-5">
      <h2 className="text-sm font-semibold text-herbrain-dark uppercase tracking-wide mb-3">
        MRI Scan
      </h2>
      
      {/* MRI Display */}
      <div ref={containerRef} className="relative flex-1 flex items-center justify-center bg-herbrain-dark rounded-xl overflow-hidden min-h-[200px]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-herbrain-dark/90 z-10">
            <div className="loading-spinner"></div>
          </div>
        )}

        {error ? (
          <div className="flex flex-col items-center justify-center p-4 text-center">
            <svg className="w-8 h-8 text-herbrain-subtle/40 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-herbrain-subtle/60 text-sm">{error}</p>
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
            <p className="text-herbrain-subtle/40 text-sm">Loading MRI...</p>
          </div>
        )}
        
        {/* View buttons - overlaid on MRI */}
        <div 
          className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 p-1 rounded-lg"
          style={{ background: 'rgba(255, 255, 255, 0.15)', backdropFilter: 'blur(8px)' }}
        >
          {(['sagittal', 'coronal', 'axial'] as ViewType[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
                view === v 
                  ? 'bg-herbrain-green text-white shadow-sm' 
                  : 'text-white/80 hover:bg-white/20'
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Slice Slider */}
      <div className="mt-3 sm:mt-4 px-1">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs text-herbrain-muted">Slice</span>
          <span className="text-xs sm:text-sm text-herbrain-dark tabular-nums font-medium">
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
