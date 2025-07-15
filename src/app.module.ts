import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AppController } from './app.controller';
import { ConfigModule } from '@nestjs/config';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { ModelsService } from './models/models.service';
import { ModelsModule } from './models/models.module';
import { ModelsController } from './models/models.controller';
import { DatasetsModule } from './datasets/datasets.module';  

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true, envFilePath: '.env' }),
    MongooseModule.forRoot(process.env.MONGODB_URI || 'mongodb://localhost:27017/trainify'),
    AuthModule,
    UsersModule,
    ModelsModule,
    DatasetsModule
  ],
  controllers: [AppController, ModelsController],
  providers: [AppService, ModelsService],
})
export class AppModule {}
