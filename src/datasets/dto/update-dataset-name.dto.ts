import { IsNotEmpty, IsString, MinLength, MaxLength } from 'class-validator';

export class UpdateDatasetNameDto {
    @IsNotEmpty({ message: 'Dataset name is required' })
    @IsString({ message: 'Dataset name must be a string' })
    @MinLength(1, { message: 'Dataset name must be at least 1 character long' })
    @MaxLength(100, { message: 'Dataset name must not exceed 100 characters' })
    name: string;
}
