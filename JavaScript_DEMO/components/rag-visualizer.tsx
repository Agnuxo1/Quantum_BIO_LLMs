'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { ColorRAGMemory } from '@/lib/rag-memory'
import { HexColorPicker } from 'react-colorful'

export function RAGVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ragMemoryRef = useRef<ColorRAGMemory>()
  const [selectedColor, setSelectedColor] = useState('#ff0000')
  const [radius, setRadius] = useState(30)
  const [isDrawing, setIsDrawing] = useState(false)
  const [tokenStats, setTokenStats] = useState<{ totalTokens: number; frequencies: { [key: string]: number } }>({
    totalTokens: 0,
    frequencies: {}
  })
  const renderTimeoutRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    ragMemoryRef.current = new ColorRAGMemory(400, 400)
    ragMemoryRef.current.generateEntangledPattern(Math.random() * 1000)
    renderMemoryBuffer()
    updateTokenStats()

    return () => {
      if (renderTimeoutRef.current) {
        clearTimeout(renderTimeoutRef.current)
      }
    }
  }, [])

  const renderMemoryBuffer = useCallback(() => {
    if (!canvasRef.current || !ragMemoryRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const buffer = ragMemoryRef.current.getMemoryBuffer()
    const imageData = ctx.createImageData(400, 400)
    
    for (let i = 0; i < buffer.length; i++) {
      imageData.data[i] = buffer[i] * 255
    }
    
    ctx.putImageData(imageData, 0, 0)
  }, [])

  const updateTokenStats = useCallback(() => {
    if (!ragMemoryRef.current) return
    setTokenStats(ragMemoryRef.current.getTokenStats())
  }, [])

  const hexToRgb = useCallback((hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result ? {
      r: parseInt(result[1], 16) / 255,
      g: parseInt(result[2], 16) / 255,
      b: parseInt(result[3], 16) / 255
    } : null
  }, [])

  const handleCanvasInteraction = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !ragMemoryRef.current || !isDrawing) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const x = Math.floor((e.clientX - rect.left) * (400 / rect.width))
    const y = Math.floor((e.clientY - rect.top) * (400 / rect.height))
    
    const color = hexToRgb(selectedColor)
    if (!color) return

    ragMemoryRef.current.applyInterference(x, y, radius, color)
    
    if (renderTimeoutRef.current) {
      clearTimeout(renderTimeoutRef.current)
    }
    renderTimeoutRef.current = setTimeout(() => {
      renderMemoryBuffer()
      updateTokenStats()
    }, 16)
  }, [isDrawing, selectedColor, radius, hexToRgb, renderMemoryBuffer, updateTokenStats])

  const resetMemory = useCallback(() => {
    if (!ragMemoryRef.current) return
    ragMemoryRef.current.generateEntangledPattern(Math.random() * 1000)
    renderMemoryBuffer()
    updateTokenStats()
  }, [renderMemoryBuffer, updateTokenStats])

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Quantum RAG Memory Visualization</CardTitle>
        <CardDescription>
          Explore the holographic color mixing system with quantum-inspired tokenization
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Color Selection</label>
              <HexColorPicker color={selectedColor} onChange={setSelectedColor} />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium">Interference Radius</label>
              <Slider
                value={[radius]}
                onValueChange={([value]) => setRadius(value)}
                min={5}
                max={100}
                step={1}
              />
              <span className="text-sm text-muted-foreground">
                Current radius: {radius}px
              </span>
            </div>
            <Button onClick={resetMemory} variant="outline">
              Reset Pattern
            </Button>
          </div>
          <div className="col-span-2">
            <div className="space-y-4">
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                className="border rounded cursor-crosshair w-full"
                onMouseDown={() => setIsDrawing(true)}
                onMouseUp={() => setIsDrawing(false)}
                onMouseLeave={() => setIsDrawing(false)}
                onMouseMove={handleCanvasInteraction}
              />
              <div className="p-4 border rounded bg-muted/50">
                <h3 className="font-medium mb-2">Token Statistics</h3>
                <div className="text-sm">
                  <p>Total Unique Colors: {tokenStats.totalTokens}</p>
                  <div className="mt-2 max-h-32 overflow-y-auto">
                    {Object.entries(tokenStats.frequencies)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 10)
                      .map(([tokenId, frequency]) => (
                        <div key={tokenId} className="flex justify-between">
                          <span>Token {tokenId}</span>
                          <span>Used {frequency} times</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

