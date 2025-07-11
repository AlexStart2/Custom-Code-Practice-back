import { Controller, Get } from '@nestjs/common';
import { ModelsService } from './models.service';

@Controller('models')
export class ModelsController {
    constructor(private readonly modelsService: ModelsService) {}

    @Get('available-models')
    async getAvailableModels(): Promise<string[]> {
        return this.modelsService.getAvailableModels()
            .then(models => {
                return models;
            })
            .catch(err => {
                console.error('Error fetching available models:', err);
                throw new Error('Failed to fetch available models');
            });
    }
}
