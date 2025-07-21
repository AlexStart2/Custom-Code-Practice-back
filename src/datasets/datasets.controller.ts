import { Body, Controller, Delete, Get, Param, Patch, Req, UseGuards } from '@nestjs/common';
import { DatasetsService } from './datasets.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';


@UseGuards(JwtAuthGuard)
@Controller('datasets')
export class DatasetsController {
    constructor(private readonly datasetsService: DatasetsService) {}


    @Get('get-user-datasets')
    async getAllDatasets(@Req() req) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.getAllDatasets(user.userId);
    }

    @Delete('dataset/:id')
    async deleteDataset(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.deleteDataset(user.userId, id);
    }

    @Get('dataset/:id')
    async getDatasetById(@Req() req, @Param('id') id: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.getDatasetById(user.userId, id);
    }

    @Get('jobs')
    async getJobByUserId(@Req() req) {
        const user = req.user as { userId: string; email: string };;
        return this.datasetsService.getJobByUserId(user.userId);
    }

    @Patch('dataset/name/:id')
    async updateDatasetName(@Req() req, @Param('id') id: string, @Body('name') name: string) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.updateDatasetName(user.userId, id, name);
    }

    @Delete('/:id/files/:fileId/chunks/:idx')
    async deleteChunk(@Req() req, @Param('id') id: string, @Param('fileId') fileId: string, @Param('idx') idx: number) {
        const user = req.user as { userId: string; email: string };
        return this.datasetsService.deleteChunk(user.userId, id, fileId, idx);
    }

}
