import { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';
import WeekSlider from './WeekSlider';
import {
  loadMeshData,
  getFigureForWeek,
  createMeshLayout,
  type PrerenderedMeshes,
} from '../lib/meshData';

interface MeshExplorerProps {
  initialWeek?: number;
  onWeekChange?: (week: number) => void;
}

export default function MeshExplorer({
  initialWeek = 15,
  onWeekChange,
}: MeshExplorerProps) {
  const [meshData, setMeshData] = useState<PrerenderedMeshes | null>(null);
  const [week, setWeek] = useState(initialWeek);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showBrain, setShowBrain] = useState(true);
  const plotRef = useRef<any>(null);

  // Load mesh data on mount
  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const data = await loadMeshData();
        setMeshData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load mesh data:', err);
        setError('Failed to load brain visualization data');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Sync with external week changes
  useEffect(() => {
    setWeek(initialWeek);
  }, [initialWeek]);

  // Handle week change
  const handleWeekChange = (newWeek: number) => {
    setWeek(newWeek);
    onWeekChange?.(newWeek);
  };

  // Get Plotly data for current week (directly from prerendered figures)
  const getPlotData = (): Data[] => {
    if (!meshData) return [];

    const figure = getFigureForWeek(meshData, week);
    if (!figure || !figure.data) return [];

    // Return the prerendered Plotly data directly
    return figure.data;
  };

  // Merge prerendered layout with our custom layout
  const getLayout = (): Partial<Layout> => {
    const baseLayout = createMeshLayout(false);
    
    if (meshData) {
      const figure = getFigureForWeek(meshData, week);
      if (figure?.layout) {
        // Merge prerendered layout with our base layout
        return {
          ...figure.layout,
          ...baseLayout,
          width: 600,
          height: 500,
        };
      }
    }
    
    return {
      ...baseLayout,
      width: 600,
      height: 500,
    };
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 bg-white rounded-xl border border-gray-200 p-8">
        <div className="loading-spinner mb-4"></div>
        <p className="text-herbrain-muted">Loading brain visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 bg-white rounded-xl border border-gray-200 p-8">
        <p className="text-red-500 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-herbrain-green text-white rounded-lg hover:bg-herbrain-green/90"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center bg-white rounded-xl border border-gray-200 p-6">
      {/* 3D Mesh Plot */}
      <div className="mb-6">
        <Plot
          ref={plotRef}
          data={getPlotData()}
          layout={getLayout()}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            displaylogo: false,
            responsive: true,
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      {/* Controls */}
      <div className="w-full max-w-md space-y-4">
        {/* Week Slider */}
        <WeekSlider value={week} onChange={handleWeekChange} />

        {/* Toggle for whole brain view */}
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            id="show-brain"
            checked={showBrain}
            onChange={(e) => setShowBrain(e.target.checked)}
            className="w-4 h-4 text-herbrain-green rounded focus:ring-herbrain-green"
          />
          <label htmlFor="show-brain" className="text-sm text-herbrain-dark">
            Show Full Brain Overlay
          </label>
        </div>
      </div>

      {/* Color Legend */}
      <div className="mt-6 flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500"></div>
          <span className="text-herbrain-muted">Shrinking</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-blue-500"></div>
          <span className="text-herbrain-muted">Growing</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E5D4C0' }}></div>
          <span className="text-herbrain-muted">Unchanged</span>
        </div>
      </div>
    </div>
  );
}
