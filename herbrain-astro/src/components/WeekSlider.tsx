import { useState, useEffect, useMemo } from 'react';

interface WeekSliderProps {
  value: number;
  onChange: (week: number) => void;
  min?: number;
  max?: number;
  label?: string;
}

export default function WeekSlider({
  value,
  onChange,
  min = 0,
  max = 40,
  label = 'Gestational Week',
}: WeekSliderProps) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10);
    setLocalValue(newValue);
    onChange(newValue);
  };

  // Calculate trimester
  const trimester = useMemo(() => {
    if (localValue <= 12) return { num: 1, label: 'First Trimester' };
    if (localValue <= 27) return { num: 2, label: 'Second Trimester' };
    return { num: 3, label: 'Third Trimester' };
  }, [localValue]);

  // Calculate progress percentage
  const progress = ((localValue - min) / (max - min)) * 100;

  return (
    <div className="w-full">
      {/* Header with week number and trimester */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-baseline gap-3">
          <span className="text-sm font-medium text-herbrain-muted uppercase tracking-wide">{label}</span>
          <div className="flex items-baseline gap-1">
            <span className="text-4xl font-semibold text-herbrain-dark tabular-nums">
              {localValue}
            </span>
            <span className="text-lg text-herbrain-muted/60">/ {max}</span>
          </div>
        </div>
        
        <span className="pill-badge pill-badge-green text-sm">
          {trimester.label}
        </span>
      </div>
      
      {/* Slider container */}
      <div className="relative pt-2 pb-1">
        {/* Track background with trimester markers */}
        <div className="relative h-2 bg-herbrain-surface rounded-full">
          {/* Progress fill */}
          <div 
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-herbrain-green to-herbrain-green-light rounded-full transition-all duration-75"
            style={{ width: `${progress}%` }}
          />
          
          {/* Trimester markers */}
          <div 
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-herbrain-border/80"
            style={{ left: '30%' }}
          />
          <div 
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-herbrain-border/80"
            style={{ left: '67.5%' }}
          />
        </div>
        
        {/* Range input - styled */}
        <input
          type="range"
          min={min}
          max={max}
          value={localValue}
          onChange={handleChange}
          className="absolute inset-0 w-full opacity-0 cursor-pointer z-10"
          style={{ height: '28px', top: '-6px' }}
        />
        
        {/* Custom thumb */}
        <div 
          className="absolute top-1/2 -translate-y-1/2 pointer-events-none transition-all duration-75"
          style={{ left: `calc(${progress}% - 10px)` }}
        >
          <div className="w-5 h-5 rounded-full bg-white border-[3px] border-herbrain-green shadow-md" />
        </div>
      </div>
      
      {/* Week labels */}
      <div className="flex justify-between text-sm text-herbrain-muted mt-3 px-0.5">
        <span>Week 0</span>
        <span>Week 12</span>
        <span>Week 27</span>
        <span>Week 40</span>
      </div>
    </div>
  );
}
