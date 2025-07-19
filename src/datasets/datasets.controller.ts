import { Body, Controller, Delete, Get, Param, Patch, Req, UseGuards } from '@nestjs/common';
import { DatasetsService } from './datasets.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';


@Controller('datasets')
export class DatasetsController {
    constructor(private readonly datasetsService: DatasetsService) {}


    @UseGuards(JwtAuthGuard)
    @Get('get-user-datasets')
    async getAllDatasets(@Req() req) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.getAllDatasets(user.userId);
    }

    @UseGuards(JwtAuthGuard)
    @Delete('dataset/:id')
    async deleteDataset(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.deleteDataset(user.userId, id);
    }

    @UseGuards(JwtAuthGuard)
    @Get('dataset/:id')
    async getDatasetById(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.getDatasetById(user.userId, id);
    }

    @UseGuards(JwtAuthGuard)
    @Get('jobs')
    async getJobByUserId(@Req() req) {
        const user = req.user as { userId: string; email: string };;
        return this.datasetsService.getJobByUserId(user.userId);
    }

    @UseGuards(JwtAuthGuard)
    @Patch('dataset/name/:id')
    async updateDatasetName(@Req() req, @Param('id') id: string, @Body('name') name: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.updateDatasetName(user.userId, id, name);
    }

}
