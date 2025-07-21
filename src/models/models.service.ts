import { Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface OllamaModel {
    name: string;
    size: string;
    modified: string;
    digest?: string;
}

@Injectable()
export class ModelsService {
    constructor() {}

    async getAvailableModels(): Promise<OllamaModel[]> {
        try {
            // Use promisified exec for better error handling and async/await
            const { stdout, stderr } = await execAsync('ollama list');
            
            if (stderr) {
                console.error(`Ollama command stderr: ${stderr}`);
                throw new InternalServerErrorException('Failed to execute ollama command');
            }

            const lines = stdout.split('\n').filter(line => line.trim() !== '');
            
            if (lines.length <= 1) {
                return []; // No models available
            }

            // Skip header line and parse model information
            const modelData = lines.slice(1)
                .map(line => {
                    const parts = line.split(/\s+/);
                    if (parts.length >= 3) {
                        return {
                            name: parts[0],
                            size: parts[1],
                            modified: parts[2],
                            digest: parts[3] || undefined
                        } as OllamaModel;
                    }
                    return null;
                })
                .filter((model): model is OllamaModel => model !== null);

            return modelData;
            
        } catch (error) {
            console.error('Error fetching available models:', error);
            
            // Check if ollama is installed/available
            if (error.message?.includes('command not found') || error.message?.includes('ollama')) {
                throw new InternalServerErrorException('Ollama is not installed or not available in PATH');
            }
            
            throw new InternalServerErrorException('Failed to fetch available models');
        }
    }

    async getModelNames(): Promise<string[]> {
        const models = await this.getAvailableModels();
        return models.map(model => model.name);
    }

    async checkModelExists(modelName: string): Promise<boolean> {
        try {
            const models = await this.getModelNames();
            return models.includes(modelName);
        } catch (error) {
            console.error('Error checking model existence:', error);
            return false;
        }
    }

    async getModelInfo(modelName: string): Promise<OllamaModel> {
        const models = await this.getAvailableModels();
        const model = models.find(m => m.name === modelName);
        
        if (!model) {
            throw new NotFoundException(`Model '${modelName}' not found`);
        }
        
        return model;
    }
}
