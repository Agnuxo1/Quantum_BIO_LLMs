'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { Play, Pause, RotateCcw } from 'lucide-react'
import dynamic from 'next/dynamic'

const ForceGraph = dynamic(() => import('react-force-graph-2d'), { ssr: false })

export default function NeuralDemo() {
  const [isRunning, setIsRunning] = useState(false)
  const [neurons, setNeurons] = useState(50)
  const [activity, setActivity] = useState(0)
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const graphRef = useRef<any>(null)

  const generateGraph = useCallback(() => {
    const nodes = Array.from({ length: neurons }, (_, i) => ({
      id: `node${i}`,
      val: Math.random() * 10
    }))

    const links = []
    for (let i = 0; i < neurons; i++) {
      const numLinks = Math.floor(Math.random() * 3) + 1
      for (let j = 0; j < numLinks; j++) {
        const target = Math.floor(Math.random() * neurons)
        if (target !== i) {
          links.push({
            source: `node${i}`,
            target: `node${target}`,
            value: Math.random()
          })
        }
      }
    }

    setGraphData({ nodes, links })
  }, [neurons])

  useEffect(() => {
    generateGraph()
  }, [neurons, generateGraph])

  const startSimulation = useCallback(() => {
    if (!isRunning) {
      setIsRunning(true)
      intervalRef.current = setInterval(() => {
        setActivity(prev => (prev + Math.random() * 10) % 100)
        generateGraph()
      }, 1000)
    }
  }, [isRunning, generateGraph])

  const stopSimulation = useCallback(() => {
    if (isRunning) {
      setIsRunning(false)
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isRunning])

  const resetSimulation = useCallback(() => {
    stopSimulation()
    setActivity(0)
    generateGraph()
    if (graphRef.current) {
      graphRef.current.centerAt(600, 200, 1000)
      graphRef.current.zoom(1, 1000)
    }
  }, [stopSimulation, generateGraph])

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Neural Architecture Simulation</CardTitle>
          <CardDescription>
            Visualize the bioinspired neural network in action
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Number of Neurons</label>
            <Slider
              value={[neurons]}
              onValueChange={([value]) => setNeurons(value)}
              min={10}
              max={100}
              step={1}
            />
            <span className="text-sm text-muted-foreground">
              Current: {neurons} neurons
            </span>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Neural Activity</label>
            <Progress value={activity} />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>0%</span>
              <span>{activity.toFixed(1)}%</span>
              <span>100%</span>
            </div>
          </div>

          <div className="h-[400px] w-full border rounded overflow-hidden">
            <ForceGraph
              ref={graphRef}
              graphData={graphData}
              nodeRelSize={6}
              linkWidth={1}
              linkDirectionalParticles={2}
              linkDirectionalParticleSpeed={0.01}
              cooldownTicks={100}
              onEngineStop={() => {}}
              width={1200}
              height={400}
            />
          </div>

          <div className="flex gap-4">
            <Button onClick={isRunning ? stopSimulation : startSimulation}>
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

