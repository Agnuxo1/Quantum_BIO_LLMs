'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { Play, Pause, RotateCcw } from 'lucide-react'

export default function QuantumDemo() {
  const [isRunning, setIsRunning] = useState(false)
  const [qubits, setQubits] = useState(3)
  const [progress, setProgress] = useState(0)
  const [optimization, setOptimization] = useState(0)

  const toggleSimulation = () => {
    setIsRunning(prevIsRunning => {
      if (!prevIsRunning) {
        const interval = setInterval(() => {
          setProgress(p => {
            if (p >= 100) {
              clearInterval(interval)
              setIsRunning(false)
              return 100
            }
            return p + 1
          })
          setOptimization(prevOptimization => Math.sin(prevOptimization / 10) * 50 + 50)
        }, 100)
      }
      return !prevIsRunning
    })
  }

  const resetSimulation = () => {
    setIsRunning(false)
    setProgress(0)
    setOptimization(0)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Quantum Optimization Simulation</CardTitle>
          <CardDescription>
            Visualize how quantum processing optimizes language model performance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Number of Qubits</label>
            <Slider
              value={[qubits]}
              onValueChange={([value]) => setQubits(value)}
              min={1}
              max={10}
              step={1}
            />
            <span className="text-sm text-muted-foreground">
              Current: {qubits} qubits
            </span>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Optimization Progress</label>
            <Progress value={progress} />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>0%</span>
              <span>{progress.toFixed(1)}%</span>
              <span>100%</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Current Optimization</label>
            <div className="text-3xl font-bold">
              {optimization.toFixed(2)}%
            </div>
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

