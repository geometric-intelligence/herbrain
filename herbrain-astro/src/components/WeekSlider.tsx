import { useState, useEffect, useMemo } from 'react';

interface WeekSliderProps {
  value: number;
  onChange: (week: number) => void;
  min?: number;
  max?: number;
  label?: string;
  hideWeek40?: boolean;
  compact?: boolean;
}

export default function WeekSlider({
  value,
  onChange,
  min = 0,
  max = 40,
  label = 'Gestational Week',
  hideWeek40 = false,
  compact = false,
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
      {/* Header with week number and trimester - all on one line when compact */}
      <div className={`flex items-center justify-between gap-2 ${compact ? 'mb-1.5' : 'mb-2 sm:mb-3'}`}>
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span className={`font-medium text-herbrain-muted uppercase tracking-wide ${compact ? 'text-[9px]' : 'text-[10px] sm:text-xs'}`}>{label}</span>
          <div className="flex items-baseline gap-0.5">
            <span className={`font-semibold text-herbrain-dark tabular-nums ${compact ? 'text-base' : 'text-xl sm:text-2xl'}`}>
              {localValue}
            </span>
            <span className={`text-herbrain-muted/60 ${compact ? 'text-[10px]' : 'text-xs sm:text-sm'}`}>/{max}</span>
          </div>
        </div>
        <span className={`pill-badge pill-badge-green ${compact ? 'text-[9px] py-0.5 px-1.5' : 'text-[10px] sm:text-xs py-0.5 px-2 sm:py-1 sm:px-2.5'}`}>
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
      <div className={`flex justify-between text-herbrain-muted px-0.5 ${compact ? 'text-[9px] mt-1' : 'text-[10px] sm:text-xs mt-2'}`}>
        <span>0</span>
        <span className="hidden sm:inline">Week 12</span>
        <span className="sm:hidden">12</span>
        <span className="hidden sm:inline">Week 27</span>
        <span className="sm:hidden">27</span>
        {!hideWeek40 && (
          <>
            <span className="hidden sm:inline">Week 40</span>
            <span className="sm:hidden">40</span>
          </>
        )}
      </div>
    </div>
  );
}
