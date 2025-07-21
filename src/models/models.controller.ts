import { Controller, Get, UseGuards } from '@nestjs/common';
import { ModelsService } from './models.service';
import { JwtAuthGuard } from 'src/auth/jwt-auth.guard';

@UseGuards(JwtAuthGuard)
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
