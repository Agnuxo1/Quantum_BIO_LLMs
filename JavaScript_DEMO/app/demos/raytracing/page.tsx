'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Play, Pause, RotateCcw } from 'lucide-react'

export default function RayTracingDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [complexity, setComplexity] = useState(5)
  const frameRef = useRef(0)
  const animationRef = useRef<number>()

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const imageData = ctx.createImageData(width, height)

    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const i = (y * width + x) * 4
        const t = frameRef.current * 0.01

        const r = Math.sin(x * complexity * 0.01 + t) * 127 + 128
        const g = Math.sin(y * complexity * 0.01 + t) * 127 + 128
        const b = Math.sin((x + y) * complexity * 0.01 + t) * 127 + 128

        imageData.data[i] = r
        imageData.data[i + 1] = g
        imageData.data[i + 2] = b
        imageData.data[i + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)

    if (isRunning) {
      frameRef.current++
      animationRef.current = requestAnimationFrame(render)
    }
  }, [complexity, isRunning])

  useEffect(() => {
    render()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [render])

  const toggleSimulation = useCallback(() => {
    setIsRunning(prev => !prev)
  }, [])

  const resetSimulation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
    setIsRunning(false)
    frameRef.current = 0
    render()
  }, [render])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Ray Tracing Visualization</CardTitle>
          <CardDescription>
            Explore an interactive ray tracing simulation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Complexity</label>
            <Slider
              value={[complexity]}
              onValueChange={([value]) => setComplexity(value)}
              min={1}
              max={10}
              step={1}
            />
            <span className="text-sm text-muted-foreground">
              Current complexity: {complexity}
            </span>
          </div>

          <div className="border rounded">
            <canvas
              ref={canvasRef}
              width={400}
              height={400}
              className="w-full h-[400px]"
            />
          </div>

          <div className="flex gap-4">
            <Button onClick={toggleSimulation}>
              {isRunning ? (
                <>
                  <Pause className="mr-2 h-4 w-4" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Start
                </>
              )}
            </Button>
            <Button variant="outline" onClick={resetSimulation}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

