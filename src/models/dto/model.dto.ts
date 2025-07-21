import { IsString, IsNotEmpty } from 'class-validator';

export class ModelNameDto {
    @IsString()
    @IsNotEmpty()
    name: string;
}

export class ModelInfoDto {
    name: string;
    size: string;
    modified: string;
    digest?: string;
}
