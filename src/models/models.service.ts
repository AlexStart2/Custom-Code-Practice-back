import { Injectable } from '@nestjs/common';

@Injectable()
export class ModelsService {
    constructor() {}

    async getAvailableModels(): Promise<string[]> {
        try {
            // run ollama list command to get available models
            const { exec } = require('child_process');
            return new Promise((resolve, reject) => {
                exec('ollama list', (error: any, stdout: string, stderr: string) => {
                    if (error) {
                        console.error(`Error executing command: ${error.message}`);
                        reject(error);
                    } else if (stderr) {
                        console.error(`Command stderr: ${stderr}`);
                        reject(new Error(stderr));
                    } else {
                        
                        // Parse the output to extract model names
                        const models = stdout.split('\n').filter(line => line.trim() !== '').map(line => {
                            const parts = line.split(/\s+/);
                            return parts[0]; // Assuming the first part is the model name
                        });
                        resolve(models);

                    }
                });
            });
            
        } catch (error) {
            console.error('Error fetching available models:', error);
            throw new Error('Failed to fetch available models');
        }
    }
}
