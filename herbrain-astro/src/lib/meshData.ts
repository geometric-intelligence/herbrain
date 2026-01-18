/**
 * Pre-computed mesh data loader
 * 
 * This module loads the existing prerendered Plotly figures from the Dash app.
 * The data is already in Plotly format with Mesh3d traces.
 */

import type { Data, Layout } from 'plotly.js';

// Existing Plotly figure format from the Dash app
export interface PlotlyFigure {
  data: Data[];
  layout?: Partial<Layout>;
}

export interface PrerenderedMeshes {
  [week: string]: PlotlyFigure;
}

let cachedMeshData: PrerenderedMeshes | null = null;

/**
 * Load pre-computed mesh data (Plotly figures) from JSON file
 */
export async function loadMeshData(): Promise<PrerenderedMeshes> {
  if (cachedMeshData) {
    return cachedMeshData;
  }
  
  const response = await fetch('/data/prerendered_meshes.json');
  if (!response.ok) {
    throw new Error(`Failed to load mesh data: ${response.statusText}`);
  }
  
  cachedMeshData = await response.json();
  return cachedMeshData!;
}

/**
 * Get available weeks from the prerendered data
 */
export function getAvailableWeeks(meshData: PrerenderedMeshes): number[] {
  return Object.keys(meshData)
    .map(Number)
    .filter(n => !isNaN(n))
    .sort((a, b) => a - b);
}

/**
 * Get Plotly figure data for a specific gestational week
 * Returns the nearest available week if exact week is not available
 */
export function getFigureForWeek(
  meshData: PrerenderedMeshes,
  week: number
): PlotlyFigure | null {
  const availableWeeks = getAvailableWeeks(meshData);
  
  if (availableWeeks.length === 0) {
    return null;
  }
  
  // Find nearest available week
  let nearestWeek = availableWeeks[0];
  let minDiff = Math.abs(week - nearestWeek);
  
  for (const w of availableWeeks) {
    const diff = Math.abs(week - w);
    if (diff < minDiff) {
      minDiff = diff;
      nearestWeek = w;
    }
  }
  
  return meshData[String(nearestWeek)] || null;
}

/**
 * Create optimized Plotly layout for 3D mesh visualization
 */
export function createMeshLayout(showAxes: boolean = false): Partial<Layout> {
  return {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      aspectmode: 'data',
      xaxis: {
        visible: showAxes,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      yaxis: {
        visible: showAxes,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      zaxis: {
        visible: showAxes,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      bgcolor: 'rgba(250, 251, 252, 0)',
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    uirevision: 'constant',
  } as Partial<Layout>;
}

/**
 * Week to session mapping (metadata for MRI)
 */
export interface SessionMetadata {
  weekToSession: Record<number, string>;
  sessionUrls: Record<string, string>;
}

let cachedMetadata: SessionMetadata | null = null;

export async function loadSessionMetadata(): Promise<SessionMetadata> {
  if (cachedMetadata) {
    return cachedMetadata;
  }
  
  try {
    const response = await fetch('/data/metadata.json');
    if (!response.ok) {
      throw new Error('Metadata not found');
    }
    cachedMetadata = await response.json();
    return cachedMetadata!;
  } catch {
    // Return default empty metadata if file doesn't exist
    return {
      weekToSession: {},
      sessionUrls: {},
    };
  }
}

/**
 * Get session ID for a gestational week
 */
export function getSessionForWeek(
  metadata: SessionMetadata,
  week: number
): string | null {
  const availableWeeks = Object.keys(metadata.weekToSession).map(Number);
  
  if (availableWeeks.length === 0) {
    return null;
  }
  
  let nearestWeek = availableWeeks[0];
  let minDiff = Math.abs(week - nearestWeek);
  
  for (const w of availableWeeks) {
    const diff = Math.abs(week - w);
    if (diff < minDiff) {
      minDiff = diff;
      nearestWeek = w;
    }
  }
  
  return metadata.weekToSession[nearestWeek] || null;
}
