'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Loader2 } from 'lucide-react'

const CANVAS_SIZE = 256;

class HolographicMemory {
  private memory: Float32Array;

  constructor(private width: number, private height: number) {
    this.memory = new Float32Array(width * height * 4);
    this.initializeMemory();
  }

  private initializeMemory() {
    for (let i = 0; i < this.memory.length; i += 4) {
      this.memory[i] = Math.random();     // R
      this.memory[i + 1] = Math.random(); // G
      this.memory[i + 2] = Math.random(); // B
      this.memory[i + 3] = 1;             // A
    }
  }

  write(x: number, y: number, value: number) {
    const index = (y * this.width + x) * 4;
    this.memory[index] = value;
    this.memory[index + 1] = value;
    this.memory[index + 2] = value;
  }

  update() {
    for (let i = 0; i < this.memory.length; i += 4) {
      this.memory[i] = (this.memory[i] + Math.random() * 0.1) % 1;     // R
      this.memory[i + 1] = (this.memory[i + 1] + Math.random() * 0.1) % 1; // G
      this.memory[i + 2] = (this.memory[i + 2] + Math.random() * 0.1) % 1; // B
    }
  }

  getMemoryState(): Float32Array {
    return this.memory;
  }
}

export function HolographicDisplay() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const memoryRef = useRef<HolographicMemory>();
  const [isActive, setIsActive] = useState(false);
  const [intensity, setIntensity] = useState(50);
  const [isProcessing, setIsProcessing] = useState(false);
  const animationFrameRef = useRef<number>();
  const isDrawing = useRef(false);

  useEffect(() => {
    memoryRef.current = new HolographicMemory(CANVAS_SIZE, CANVAS_SIZE);
    renderFrame();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const renderFrame = useCallback(() => {
    if (!canvasRef.current || !memoryRef.current) return;

    setIsProcessing(true);
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    memoryRef.current.update();  // Actualiza el estado de la memoria

    const imageData = ctx.createImageData(CANVAS_SIZE, CANVAS_SIZE);
    const state = memoryRef.current.getMemoryState();
    
    for (let i = 0; i < state.length; i++) {
      imageData.data[i] = state[i] * 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
    setIsProcessing(false);

    if (isActive) {
      animationFrameRef.current = requestAnimationFrame(renderFrame);
    }
  }, [isActive]);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;
    handlePointerMove(e);
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current || !canvasRef.current || !memoryRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * (CANVAS_SIZE / rect.width));
    const y = Math.floor((e.clientY - rect.top) * (CANVAS_SIZE / rect.height));

    memoryRef.current.write(x, y, intensity / 100);
    renderFrame();
  }, [intensity, renderFrame]);

  const handlePointerUp = useCallback(() => {
    isDrawing.current = false;
  }, []);

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Quantum Holographic Memory Display
          {isProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="border rounded cursor-crosshair touch-none"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
            style={{ width: '100%', maxWidth: '512px', height: 'auto' }}
          />
        </div>
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Intensity</label>
            <Slider
              value={[intensity]}
              onValueChange={([value]) => setIntensity(value)}
              min={0}
              max={100}
              step={1}
              disabled={isProcessing}
            />
          </div>
          <Button
            onClick={() => {
              setIsActive(!isActive);
              if (!isActive) {
                renderFrame();
              }
            }}
            variant={isActive ? "destructive" : "default"}
            disabled={isProcessing}
          >
            {isActive ? 'Stop' : 'Start'} Animation
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

