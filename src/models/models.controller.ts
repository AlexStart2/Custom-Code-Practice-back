import { Controller, Get, UseGuards, HttpStatus, Param } from '@nestjs/common';
import { ModelsService, OllamaModel } from './models.service';
import { JwtAuthGuard } from 'src/auth/jwt-auth.guard';
import { createResponse, ApiResponse } from '../common/interfaces/api-response.interface';

@UseGuards(JwtAuthGuard)
@Controller('models')
export class ModelsController {
    constructor(private readonly modelsService: ModelsService) {}

    @Get('available')
    async getAvailableModels(): Promise<ApiResponse<OllamaModel[]>> {
        const models = await this.modelsService.getAvailableModels();
        return createResponse(models, `Found ${models.length} available models`);
    }

    @Get('names')
    async getModelNames(): Promise<ApiResponse<string[]>> {
        const names = await this.modelsService.getModelNames();
        return createResponse(names, `Found ${names.length} model names`);
    }

    @Get('check/:name')
    async checkModelExists(@Param('name') name: string): Promise<ApiResponse<{ exists: boolean; modelName: string }>> {
        const exists = await this.modelsService.checkModelExists(name);
        const data = { exists, modelName: name };
        const message = exists ? `Model '${name}' is available` : `Model '${name}' not found`;
        return createResponse(data, message);
    }

    @Get('info/:name')
    async getModelInfo(@Param('name') name: string): Promise<ApiResponse<OllamaModel>> {
        const model = await this.modelsService.getModelInfo(name);
        return createResponse(model, `Model information retrieved for '${name}'`);
    }
}
