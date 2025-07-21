import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { UsersService } from '../users/users.service';
import { SignUpDto } from './dto/authSignUp.dto';
import { LoginDto } from './dto/authLogin.dto';
import * as bcrypt from 'bcrypt';
import { JwtService } from '@nestjs/jwt';

@Injectable()
export class AuthService {
    constructor(
        private readonly usersService: UsersService,
        private readonly jwtService: JwtService
    ) {}

    async signup(signUpDto: SignUpDto) {
        const { email, password, name } = signUpDto;

        // Check if user already exists
        const existingUser = await this.usersService.findByEmail(email);
        if (existingUser) {
            throw new ConflictException('User already exists');
        }

        const saltRounds = process.env.NODE_ENV === 'production' ? 12 : 10;
        const hashedPassword = await bcrypt.hash(password, saltRounds);

        const user = await this.usersService.createUser(email, hashedPassword, name);
        
        const { password: _, ...userWithoutPassword } = user;
        return userWithoutPassword;
    }


    async login(loginDto: LoginDto) {  // Fixed parameter name
        const { email, password } = loginDto;

        const user = await this.usersService.findByEmail(email);
        if (!user || !(await bcrypt.compare(password, user.password))) {
            throw new UnauthorizedException('Invalid email or password');
        }
        
        const payload = {
            id: user._id,
            email: user.email,
            name: user.name,
            iat: Math.floor(Date.now() / 1000), // Issued at time
        };

        const accessToken = await this.jwtService.signAsync(payload);

        return {
            access_token: accessToken,
            userData: {
                id: user._id,
                email: user.email,
                name: user.name,
            },
            expiresIn: '24h'
        };
    }

    // Add token validation method
    async validateToken(token: string) {
        try {
            return await this.jwtService.verifyAsync(token);
        } catch (error) {
            throw new UnauthorizedException('Invalid token');
        }
    }
}
