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

  // Load metadata on mount
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

  // Load volume when week changes
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

        // Reset slice index to middle
        const maxSlice = getMaxSliceIndex(vol.dims, view);
        setSliceIndex(Math.floor(maxSlice / 2));
      } catch (err) {
        console.error('Failed to load MRI volume:', err);
        setError('Failed to load MRI data. Make sure R2 storage is configured.');
      } finally {
        setLoading(false);
      }
    }

    loadVolume();
  }, [week, metadata, currentSession, view]);

  // Update slice index bounds when view changes
  useEffect(() => {
    if (!volume) return;
    const maxSlice = getMaxSliceIndex(volume.dims, view);
    setSliceIndex(Math.floor(maxSlice / 2));
  }, [view, volume]);

  // Get slice data for visualization
  const getSliceData = useCallback((): number[][] | null => {
    if (!volume) return null;

    const slice = extractSlice(volume, view, sliceIndex);
    const [width, height] = getSliceDims(volume.dims, view);

    // Convert 1D array to 2D for Plotly
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

  // Plotly data
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

  // Plotly layout
  const sliceDims = volume ? getSliceDims(volume.dims, view) : [100, 100];
  const aspectRatio = sliceDims[1] / sliceDims[0];
  const baseWidth = 350;
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
    <div className="flex flex-col items-center bg-white rounded-xl border border-gray-200 p-6">
      {/* MRI Slice Display */}
      <div className="mb-4 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
            <div className="loading-spinner"></div>
          </div>
        )}

        {error ? (
          <div className="flex items-center justify-center h-64 w-80 bg-gray-100 rounded-lg">
            <p className="text-herbrain-muted text-sm text-center px-4">
              {error}
              <br />
              <span className="text-xs mt-2 block">
                Configure R2 storage to enable MRI viewing
              </span>
            </p>
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
          <div className="flex items-center justify-center h-64 w-80 bg-gray-100 rounded-lg">
            <p className="text-herbrain-muted text-sm">Loading MRI data...</p>
          </div>
        )}
      </div>

      {/* View Selection */}
      <div className="flex items-center gap-4 mb-4">
        <span className="text-sm font-medium text-herbrain-dark">MRI View:</span>
        <div className="flex gap-2">
          {(['sagittal', 'coronal', 'axial'] as ViewType[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                view === v
                  ? 'bg-herbrain-green text-white'
                  : 'bg-gray-100 text-herbrain-dark hover:bg-gray-200'
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Slice Slider */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-herbrain-dark">
            Slice
          </label>
          <span className="text-sm text-herbrain-muted">
            {sliceIndex} / {maxSlice}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={maxSlice}
          value={sliceIndex}
          onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-herbrain-green"
          disabled={!volume || loading}
        />
      </div>
    </div>
  );
}
