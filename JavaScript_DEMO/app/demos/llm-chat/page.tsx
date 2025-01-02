'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Send, Brain } from 'lucide-react'
import dynamic from 'next/dynamic'

const ForceGraph = dynamic(() => import('react-force-graph-2d'), { ssr: false })

export default function LLMChatDemo() {
  const [input, setInput] = useState('')
  const [chat, setChat] = useState([])
  const [tokenization, setTokenization] = useState([])
  const [neuralNetwork, setNeuralNetwork] = useState({ nodes: [], links: [] })
  const [holographicMemory, setHolographicMemory] = useState([])
  const canvasRef = useRef(null)

  useEffect(() => {
    generateNeuralNetwork()
    if (canvasRef.current) {
      generateHolographicMemory()
    }
  }, [])

  const generateNeuralNetwork = () => {
    setNeuralNetwork(prevNetwork => {
      const nodes = Array.from({ length: 50 }, (_, i) => ({
        id: `node${i}`,
        val: Math.random() * 10
      }))

      const links = []
      for (let i = 0; i < 50; i++) {
        const numLinks = Math.floor(Math.random() * 3) + 1
        for (let j = 0; j < numLinks; j++) {
          const target = Math.floor(Math.random() * 50)
          if (target !== i) {
            links.push({
              source: `node${i}`,
              target: `node${target}`,
              value: Math.random()
            })
          }
        }
      }

      return { nodes, links }
    })
  }

  const generateHolographicMemory = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height
    const imageData = ctx.createImageData(width, height)

    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const i = (y * width + x) * 4
        const r = Math.sin(x * 0.1) * 127 + 128
        const g = Math.sin(y * 0.1) * 127 + 128
        const b = Math.sin((x + y) * 0.1) * 127 + 128

        imageData.data[i] = r
        imageData.data[i + 1] = g
        imageData.data[i + 2] = b
        imageData.data[i + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
    setHolographicMemory(Array.from(imageData.data))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return

    // Simulated LLM response
    const response = `This is a simulated response to: "${input}"`
    setChat(prevChat => [...prevChat, { role: 'user', content: input }, { role: 'assistant', content: response }])
    
    // Simulated tokenization
    setTokenization(input.split(' ').map(word => ({ word, value: Math.random() })))
    
    // Update neural network
    generateNeuralNetwork()
    
    // Update holographic memory
    generateHolographicMemory()

    setInput('')
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>LLM Chat with Visualization</CardTitle>
          <CardDescription>
            Interact with the LLM and visualize the underlying processes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-4 gap-4">
            <div className="col-span-3 space-y-4">
              <div className="border rounded p-4 h-[300px] overflow-y-auto">
                {chat.map((message, index) => (
                  <div key={index} className={`mb-2 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                    <span className={`inline-block p-2 rounded ${message.role === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                      {message.content}
                    </span>
                  </div>
                ))}
              </div>
              <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message..."
                />
                <Button type="submit">
                  <Send className="h-4 w-4" />
                </Button>
              </form>
            </div>
            <div className="space-y-4">
              <div className="border rounded p-4 h-[200px] overflow-y-auto">
                <h3 className="font-semibold mb-2">Tokenization</h3>
                <div className="flex flex-wrap gap-2">
                  {tokenization.map((token, index) => (
                    <span key={index} className="inline-block px-2 py-1 bg-gray-200 rounded text-sm">
                      {token.word} ({token.value.toFixed(2)})
                    </span>
                  ))}
                </div>
              </div>
              <div className="border rounded h-[200px]">
                <h3 className="font-semibold p-2">Neural Network</h3>
                <ForceGraph
                  graphData={neuralNetwork}
                  nodeRelSize={6}
                  linkWidth={1}
                  height={168}
                  width={332}
                />
              </div>
            </div>
          </div>
          <div className="border rounded p-4">
            <h3 className="font-semibold mb-2">Holographic Memory Visualization</h3>
            <canvas
              ref={canvasRef}
              width={400}
              height={100}
              className="w-full h-[100px]"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

