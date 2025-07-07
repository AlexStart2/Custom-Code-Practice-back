import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { DEFAULT_ORIGIN, DEFAULT_PORT } from './app.config';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({origin: DEFAULT_ORIGIN});

  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    transform: true,
  })
);


  await app.listen(process.env.PORT ?? DEFAULT_PORT);
}
bootstrap();
