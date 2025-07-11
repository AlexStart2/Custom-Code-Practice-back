import { Body, Controller, Post, Req, UseGuards } from '@nestjs/common';
import { DatasetsService } from './datasets.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';


@Controller('datasets')
export class DatasetsController {
    constructor(private readonly datasetsService: DatasetsService) {}



    @UseGuards(JwtAuthGuard)               // ← protect with JWT
    @Post('store-rag')
    async storeRagFiles(
    @Req() req,                          // ← grab the request
    @Body() payload: { data: any[] },
  ) {
    // req.user was populated by JwtStrategy.validate()
    const user = req.user as { userId: string; email: string };
    console.log('User ID:', user.userId);

    // pass the userId into your service
    return this.datasetsService.storeRagFiles(payload.data, user.userId);
  }
}