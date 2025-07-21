// src/auth/jwt.strategy.ts
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, ExtractJwt } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(private configService: ConfigService) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: process.env.JWT_SECRET
    });
  }

  // This runs after the token is verified; `payload` is the decoded JWT
  async validate(payload: any) {
    // Add payload validation
    if (!payload.id || !payload.email) {
      throw new UnauthorizedException('Invalid token payload');
    }

    return { 
      userId: payload.id, 
      email: payload.email,
      name: payload.name 
    };
  }
}
