import { Controller, UseGuards, Get, Req } from '@nestjs/common';
import { JwtAuthGuard } from 'src/auth/jwt-auth.guard';
import { UsersService } from './users.service';
import {History} from './schemas/user.schema';

@UseGuards(JwtAuthGuard)
@Controller('users')
export class UsersController {
    constructor(private usersService: UsersService) {}

    @Get('rag-query-history')
    async getRagQueryHistory(@Req() req): Promise<History | null> {
        const user = req.user as { userId: string; email: string };
        return this.usersService.getRagQueryHistory(user.userId);
    }
}
