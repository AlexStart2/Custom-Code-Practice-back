import { Body, Controller, Get, Req, UseGuards } from '@nestjs/common';
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
}
