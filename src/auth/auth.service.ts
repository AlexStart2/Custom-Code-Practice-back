import { Injectable, UnauthorizedException, ConflictException, HttpException } from '@nestjs/common';
import { UsersService } from '../users/users.service';
import { SignUpDto } from './dto/authSignUp.dto';
import { LoginDto } from './dto/authLogin.dto';
import * as bcrypt from 'bcrypt';
import { JwtService } from '@nestjs/jwt';

@Injectable()
export class AuthService {
    constructor(
        private readonly usersService: UsersService,
        private jwtService: JwtService
    ) {}

    async signup(signUpDto: SignUpDto) {
        const { email, password, name } = signUpDto;

        // Check if user already exists
        const existingUser = await this.usersService.findByEmail(email);
        if (existingUser) {
            throw new ConflictException('User already exists');
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 12);

        return this.usersService.createUser(email, hashedPassword, name);
    }


    async login(LoginDto: LoginDto){
        const { email, password } = LoginDto;

        const user = await this.usersService.findByEmail(email);
        if(!user || !(await bcrypt.compare(password, user.password))) {
            throw new UnauthorizedException('Invalid email or password');
        }
        
        const payload = {
            // id: user._id,
            email: user.email,
            name: user.name,
        };

        return {
            access_token: await this.jwtService.signAsync(payload),
            user: payload,
        };
    }
}
