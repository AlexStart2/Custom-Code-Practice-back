import { IsEmail, IsNotEmpty, IsString, MinLength, Matches } from "class-validator";

export class SignUpDto {
    @IsNotEmpty()
    @IsString()
    @MinLength(2, { message: 'Name must be at least 2 characters long' })
    name: string;

    @IsNotEmpty()
    @IsString()
    @IsEmail({}, { message: 'Please provide a valid email address' })
    email: string;

    @IsNotEmpty()
    @IsString()
    @MinLength(4, { message: 'Password must be at least 4 characters long' })
    @Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, {
        message: 'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    })
    password: string;
}