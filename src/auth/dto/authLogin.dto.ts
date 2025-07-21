import { IsEmail, IsNotEmpty, IsString } from "class-validator";

export class LoginDto {
    @IsNotEmpty()
    @IsString()
    @IsEmail({}, { message: 'Please provide a valid email address' })
    email: string;

    @IsNotEmpty()
    @IsString()
    password: string;
}