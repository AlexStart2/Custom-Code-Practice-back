import { Controller, Post, Body, UsePipes, ValidationPipe } from '@nestjs/common';
import { AuthService } from './auth.service';
import { SignUpDto } from './dto/authSignUp.dto'
import { LoginDto } from './dto/authLogin.dto';


// @ApiTags('auth')
// @Public()
@Controller('auth')
export class AuthController {
    constructor(private readonly authService: AuthService) {}

    @Post('login')
    async login(@Body() LoginDto: LoginDto) {
        return this.authService.login(LoginDto);
    }

    @Post('signup')
    async signup(@Body() signUpDto: SignUpDto) {
        return this.authService.signup(signUpDto);
    }
}

