// src/auth/jwt.strategy.ts
import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, ExtractJwt } from 'passport-jwt';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,               // reject expired tokens
      secretOrKey: process.env.JWT_SECRET,   // your shared secret
    });
  }

  // This runs after the token is verified; `payload` is the decoded JWT
  async validate(payload: any) {
    // `payload` typically contains `{ sub: userId, email, iat, exp, ... }`
    // You can here fetch more user info or just return the payload
    console.log('JWT Payload:', payload);
    return { userId: payload.id, email: payload.email };
  }
}
