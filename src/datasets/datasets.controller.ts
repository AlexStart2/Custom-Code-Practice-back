import { Body, Controller, Delete, Get, Param, Req, UseGuards } from '@nestjs/common';
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
    @Delete(':id')
    async deleteDataset(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.deleteDataset(user.userId, id);
    }

    @UseGuards(JwtAuthGuard)
    @Get(':id')
    async getDatasetById(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.getDatasetById(user.userId, id);
    }

}
