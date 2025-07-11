import { Injectable } from '@nestjs/common';
import axios from 'axios';

@Injectable()
export class ModelsService {
    constructor() {}

    async getAvailableModels(): Promise<string[]> {
        try {
            // Fetch the Ollama library page
            const response = await axios.get('https://ollama.com/library', {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            });

            const html = response.data;

            // Extract model names using regex (equivalent to grep -oP 'href="/library/\K[^"]+')
            const modelRegex = /href="\/library\/([^"]+)"/g;
            const models: string[] = [];
            let match;

            while ((match = modelRegex.exec(html)) !== null) {
                models.push(match[1]);
            }

            return models;
        } catch (error) {
            console.error('Error fetching available models:', error);
            throw new Error('Failed to fetch available models');
        }
    }
}
