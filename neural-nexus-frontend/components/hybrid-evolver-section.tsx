// components/hybrid-evolver-section.tsx
'use client';
import React, { useState, useRef, ChangeEvent, FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Loader2, Sparkles } from "lucide-react";
import { Switch } from '@/components/ui/switch';
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";
import RealTimePlot from './real-time-plot';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function HybridEvolverSection() {
    // State
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [evalChoice, setEvalChoice] = useState<'standard' | 'custom'>('standard');
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    // Hybrid-specific config
    const [modelClass, setModelClass] = useState('OCTMNIST_CNN');
    const [useFuzzy, setUseFuzzy] = useState(true);
    const [generations, setGenerations] = useState(20);
    const [populationSize, setPopulationSize] = useState(15);
    const [fuzzyInputs, setFuzzyInputs] = useState(2);
    const [fuzzyOutputs, setFuzzyOutputs] = useState(1);
    const [hyperparamMutationRate, setHyperparamMutationRate] = useState(0.1);
    const [weightMutationRate, setWeightMutationRate] = useState(0.1);
    const [fuzzyMutationRate, setFuzzyMutationRate] = useState(0.05);
    
    // Task status
    const [taskId, setTaskId] = useState<string | null>(null);
    const [taskStatus, setTaskStatus] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('');
    const [fitnessHistory, setFitnessHistory] = useState<number[]>([]);
    const [avgFitnessHistory, setAvgFitnessHistory] = useState<number[]>([]);
    const [diversityHistory, setDiversityHistory] = useState<number[]>([]);
    const [hybridInfo, setHybridInfo] = useState<any>(null);
    const [finalModelPath, setFinalModelPath] = useState<string | null>(null);
    const [fuzzySystemPath, setFuzzySystemPath] = useState<string | null>(null);
    
    // Refs
    const modelDefRef = useRef<HTMLInputElement>(null);
    const taskEvalRef = useRef<HTMLInputElement>(null);
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    
    // File handlers
    const handleModelDefFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        setModelDefFile(e.target.files?.[0] ?? null);
    };
    
    const handleTaskEvalFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        setTaskEvalFile(e.target.files?.[0] ?? null);
    };
    
    const handleEvalChoiceChange = (value: 'standard' | 'custom') => {
        setEvalChoice(value);
        if (value === 'standard' && taskEvalRef.current) {
            taskEvalRef.current.value = "";
            setTaskEvalFile(null);
        }
    };
    
    // Poll task status
    const pollTaskStatus = async (id: string) => {
        try {
            const response = await fetch(`${API_URL}/api/v1/hybrid/status/${id}`);
            const data = await response.json();
            
            setTaskStatus(data.state);
            setProgress(data.progress || 0);
            setMessage(data.message || '');
            
            if (data.fitness_history) setFitnessHistory(data.fitness_history);
            if (data.avg_fitness_history) setAvgFitnessHistory(data.avg_fitness_history);
            if (data.diversity_history) setDiversityHistory(data.diversity_history);
            if (data.hybrid_info) setHybridInfo(data.hybrid_info);
            
            if (data.state === 'SUCCESS') {
                setFinalModelPath(data.final_model_path);
                setFuzzySystemPath(data.fuzzy_system_path);
                toast.success('Hybrid evolution completed!');
                if (pollingIntervalRef.current) {
                    clearInterval(pollingIntervalRef.current);
                    pollingIntervalRef.current = null;
                }
            } else if (data.state === 'FAILURE') {
                toast.error(`Task failed: ${data.error || 'Unknown error'}`);
                if (pollingIntervalRef.current) {
                    clearInterval(pollingIntervalRef.current);
                    pollingIntervalRef.current = null;
                }
            }
        } catch (error: any) {
            console.error('Error polling task status:', error);
        }
    };
    
    // Submit handler
    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        
        if (!modelDefFile) {
            toast.error("Model Definition file is required.");
            return;
        }
        
        if (evalChoice === 'custom' && !taskEvalFile) {
            toast.error("Custom Evaluation Script file is required.");
            return;
        }
        
        // Build config
        const config = {
            generations,
            population_size: populationSize,
            model_class: modelClass,
            use_fuzzy: useFuzzy,
            fuzzy_num_inputs: fuzzyInputs,
            fuzzy_num_outputs: fuzzyOutputs,
            hyperparam_mutation_rate: hyperparamMutationRate,
            weight_mutation_rate: weightMutationRate,
            fuzzy_mutation_rate: fuzzyMutationRate,
            hyperparam_mutation_strength: 0.02,
            weight_mutation_strength: 0.05,
            fuzzy_mutation_strength: 0.01,
            elitism_count: 2,
            tournament_size: 3,
            evolvable_hyperparams: {
                dropout: {
                    range: [0.1, 0.5],
                    type: 'float'
                }
            },
            eval_config: {
                dataset: 'mnist',
                batch_size: 128,
                num_epochs: 1
            }
        };
        
        setIsSubmitting(true);
        toast("Submitting hybrid evolution task...");
        
        const formData = new FormData();
        formData.append('model_definition', modelDefFile);
        formData.append('use_standard_eval', String(evalChoice === 'standard'));
        if (evalChoice === 'custom' && taskEvalFile) {
            formData.append('task_evaluation', taskEvalFile);
        }
        formData.append('config', JSON.stringify(config));
        
        try {
            const response = await fetch(`${API_URL}/api/v1/hybrid/start`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            setTaskId(data.task_id);
            setTaskStatus('PENDING');
            toast.success(`Task ${data.task_id} started!`);
            
            // Start polling
            pollingIntervalRef.current = setInterval(() => {
                pollTaskStatus(data.task_id);
            }, 3000);
            
            // Clear files
            if (modelDefRef.current) modelDefRef.current.value = "";
            if (taskEvalRef.current) taskEvalRef.current.value = "";
            setModelDefFile(null);
            setTaskEvalFile(null);
            
        } catch (error: any) {
            console.error("Error starting hybrid evolution:", error);
            toast.error(`Failed to start task: ${error.message || 'Unknown error'}`);
        } finally {
            setIsSubmitting(false);
        }
    };
    
    // Cleanup on unmount
    React.useEffect(() => {
        return () => {
            if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
            }
        };
    }, []);
    
    const hasPlotData = fitnessHistory.length > 0 || avgFitnessHistory.length > 0;
    const isActive = taskStatus === 'PROGRESS' || taskStatus === 'STARTED' || taskStatus === 'PENDING';
    
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Configuration Card */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-purple-500" />
                        Hybrid Neuro-Fuzzy Evolution
                    </CardTitle>
                    <CardDescription>
                        Phase 1: Co-evolve neural networks with fuzzy logic systems
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* File Inputs */}
                        <div>
                            <Label htmlFor="model-def">Model Definition (.py) <span className="text-red-500">*</span></Label>
                            <Input
                                ref={modelDefRef}
                                id="model-def"
                                type="file"
                                accept=".py"
                                required
                                onChange={handleModelDefFileChange}
                                disabled={isSubmitting || isActive}
                            />
                        </div>
                        
                        <div>
                            <Label>Evaluation Method <span className="text-red-500">*</span></Label>
                            <RadioGroup
                                value={evalChoice}
                                onValueChange={handleEvalChoiceChange}
                                className="flex space-x-4 mt-1"
                                disabled={isSubmitting || isActive}
                            >
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="standard" id="eval-standard" />
                                    <Label htmlFor="eval-standard">Standard (MNIST)</Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="custom" id="eval-custom" />
                                    <Label htmlFor="eval-custom">Upload Custom</Label>
                                </div>
                            </RadioGroup>
                        </div>
                        
                        {evalChoice === 'custom' && (
                            <div>
                                <Label htmlFor="task-eval">Custom Evaluation Script (.py) <span className="text-red-500">*</span></Label>
                                <Input
                                    ref={taskEvalRef}
                                    id="task-eval"
                                    type="file"
                                    accept=".py"
                                    required={evalChoice === 'custom'}
                                    onChange={handleTaskEvalFileChange}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                        )}
                        
                        {/* Hybrid Configuration */}
                        <div className="border-t pt-4 space-y-4">
                            <h3 className="text-sm font-medium">Hybrid Configuration</h3>
                            
                            <div>
                                <Label htmlFor="model-class">Model Class Name</Label>
                                <Input
                                    id="model-class"
                                    type="text"
                                    value={modelClass}
                                    onChange={(e) => setModelClass(e.target.value)}
                                    placeholder="e.g., OCTMNIST_CNN, MyCNN"
                                    disabled={isSubmitting || isActive}
                                />
                                <p className="text-xs text-muted-foreground mt-1">
                                    Must match the class name in your model file
                                </p>
                            </div>
                            
                            <div className="flex items-center justify-between">
                                <Label htmlFor="use-fuzzy">Enable Fuzzy Component</Label>
                                <Switch
                                    id="use-fuzzy"
                                    checked={useFuzzy}
                                    onCheckedChange={setUseFuzzy}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                            
                            {useFuzzy && (
                                <div className="space-y-3 pl-4 border-l-2 border-purple-200">
                                    <div>
                                        <Label>Fuzzy Inputs: {fuzzyInputs}</Label>
                                        <Slider
                                            value={[fuzzyInputs]}
                                            onValueChange={(v) => setFuzzyInputs(v[0])}
                                            min={1}
                                            max={5}
                                            step={1}
                                            disabled={isSubmitting || isActive}
                                        />
                                    </div>
                                    <div>
                                        <Label>Fuzzy Outputs: {fuzzyOutputs}</Label>
                                        <Slider
                                            value={[fuzzyOutputs]}
                                            onValueChange={(v) => setFuzzyOutputs(v[0])}
                                            min={1}
                                            max={3}
                                            step={1}
                                            disabled={isSubmitting || isActive}
                                        />
                                    </div>
                                    <div>
                                        <Label>Fuzzy Mutation Rate: {fuzzyMutationRate.toFixed(2)}</Label>
                                        <Slider
                                            value={[fuzzyMutationRate]}
                                            onValueChange={(v) => setFuzzyMutationRate(v[0])}
                                            min={0}
                                            max={0.3}
                                            step={0.01}
                                            disabled={isSubmitting || isActive}
                                        />
                                    </div>
                                </div>
                            )}
                            
                            <div>
                                <Label>Generations: {generations}</Label>
                                <Slider
                                    value={[generations]}
                                    onValueChange={(v) => setGenerations(v[0])}
                                    min={5}
                                    max={100}
                                    step={5}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                            
                            <div>
                                <Label>Population Size: {populationSize}</Label>
                                <Slider
                                    value={[populationSize]}
                                    onValueChange={(v) => setPopulationSize(v[0])}
                                    min={5}
                                    max={50}
                                    step={5}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                            
                            <div>
                                <Label>Hyperparam Mutation Rate: {hyperparamMutationRate.toFixed(2)}</Label>
                                <Slider
                                    value={[hyperparamMutationRate]}
                                    onValueChange={(v) => setHyperparamMutationRate(v[0])}
                                    min={0}
                                    max={0.3}
                                    step={0.01}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                            
                            <div>
                                <Label>Weight Mutation Rate: {weightMutationRate.toFixed(2)}</Label>
                                <Slider
                                    value={[weightMutationRate]}
                                    onValueChange={(v) => setWeightMutationRate(v[0])}
                                    min={0}
                                    max={0.3}
                                    step={0.01}
                                    disabled={isSubmitting || isActive}
                                />
                            </div>
                        </div>
                        
                        <Button
                            type="submit"
                            className="w-full"
                            disabled={isSubmitting || isActive || !modelDefFile || (evalChoice === 'custom' && !taskEvalFile)}
                        >
                            {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            {isSubmitting ? "Submitting..." : isActive ? "Task Running..." : "Start Hybrid Evolution"}
                        </Button>
                    </form>
                </CardContent>
            </Card>
            
            {/* Status Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Task Status & Results</CardTitle>
                    <CardDescription>Monitor hybrid evolution progress</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {taskId ? (
                        <div className="space-y-3">
                            <div>
                                <p className="text-sm">Task ID: <span className="font-mono text-xs bg-muted px-1 rounded">{taskId}</span></p>
                                <p className="text-sm">Status: <span className={`font-semibold ${taskStatus === 'SUCCESS' ? 'text-green-600' : taskStatus === 'FAILURE' ? 'text-red-600' : ''}`}>{taskStatus || 'N/A'}</span></p>
                                {useFuzzy && <p className="text-sm text-purple-600">Mode: Neuro-Fuzzy Hybrid</p>}
                            </div>
                            
                            {isActive && (
                                <div>
                                    <Progress value={progress * 100} className="w-full" />
                                    <p className="text-sm text-muted-foreground mt-1">{Math.round(progress * 100)}% complete</p>
                                </div>
                            )}
                            
                            {message && <p className="text-sm text-muted-foreground">{message}</p>}
                            
                            {hybridInfo && (
                                <div className="text-xs bg-muted p-2 rounded space-y-1">
                                    <p>Hyperparams: {hybridInfo.num_hyperparams}</p>
                                    <p>NN Weights: {hybridInfo.num_nn_weights?.toLocaleString()}</p>
                                    {hybridInfo.used_fuzzy && (
                                        <p className="text-purple-600">Fuzzy Params: {hybridInfo.num_fuzzy_params}</p>
                                    )}
                                </div>
                            )}
                            
                            {taskStatus === 'SUCCESS' && finalModelPath && (
                                <div className="space-y-2">
                                    <Button variant="outline" size="sm" className="w-full" asChild>
                                        <a href={`${API_URL}/api/v1/hybrid/download/${taskId}/model`} download>
                                            Download Model
                                        </a>
                                    </Button>
                                    {fuzzySystemPath && (
                                        <Button variant="outline" size="sm" className="w-full" asChild>
                                            <a href={`${API_URL}/api/v1/hybrid/download/${taskId}/fuzzy`} download>
                                                Download Fuzzy System
                                            </a>
                                        </Button>
                                    )}
                                </div>
                            )}
                            
                            {taskStatus === 'FAILURE' && (
                                <Alert variant="destructive">
                                    <AlertCircle className="h-4 w-4" />
                                    <AlertTitle>Task Failed</AlertTitle>
                                    <AlertDescription>Check logs for details</AlertDescription>
                                </Alert>
                            )}
                        </div>
                    ) : (
                        <p className="text-muted-foreground">Submit a task to see status</p>
                    )}
                    
                    {/* Plot */}
                    <div className="mt-4 h-72 border rounded bg-muted/20 flex items-center justify-center">
                        {hasPlotData ? (
                            <RealTimePlot
                                maxFitnessData={fitnessHistory}
                                avgFitnessData={avgFitnessHistory}
                                diversityData={diversityHistory}
                            />
                        ) : (
                            <p className="text-muted-foreground">
                                {taskId ? "Plot will appear here..." : "Submit task for plot"}
                            </p>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
