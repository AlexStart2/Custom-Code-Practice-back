import { Body, Controller, Delete, Get, Param, Patch, Req, UseGuards, ParseIntPipe } from '@nestjs/common';
import { DatasetsService } from './datasets.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { UpdateDatasetNameDto } from './dto/update-dataset-name.dto';
import { AuthenticatedUser } from '../common/interfaces/user.interface';


@UseGuards(JwtAuthGuard)
@Controller('datasets')
export class DatasetsController {
    constructor(private readonly datasetsService: DatasetsService) {}

    @Get('get-user-datasets')
    async getAllDatasets(@Req() req) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.getAllDatasets(user.userId);
    }

    @Get('jobs')
    async getJobByUserId(@Req() req) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.getJobByUserId(user.userId);
    }

    @Delete('dataset/:id')
    async deleteDataset(@Req() req, @Param('id') id: string) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.deleteDataset(user.userId, id);
    }

    @Get('dataset/:id')
    async getDatasetById(@Req() req, @Param('id') id: string) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.getDatasetById(user.userId, id);
    }

    @Patch('dataset/name/:id')
    async updateDatasetName(@Req() req, @Param('id') id: string, @Body() updateDatasetNameDto: UpdateDatasetNameDto) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.updateDatasetName(user.userId, id, updateDatasetNameDto.name);
    }

    @Delete('/:id/files/:fileId/chunks/:idx')
    async deleteChunk(
        @Req() req, 
        @Param('id') id: string, 
        @Param('fileId') fileId: string, 
        @Param('idx', ParseIntPipe) idx: number
    ) {
        const user = req.user as AuthenticatedUser;
        return this.datasetsService.deleteChunk(user.userId, id, fileId, idx);
    }
}
