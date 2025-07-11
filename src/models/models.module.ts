import { Module } from '@nestjs/common';
import { ModelsService } from './models.service';
import { ModelsController } from './models.controller';

@Module({
      imports: [],
      providers: [ModelsService],
      controllers: [ModelsController],
      exports: [ModelsService]
})
export class ModelsModule {}
