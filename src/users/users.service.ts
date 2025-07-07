import { Injectable } from '@nestjs/common';
import { Model } from 'mongoose';
import { User, UserDocument } from './schemas/user.schema';
import { InjectModel } from '@nestjs/mongoose';


@Injectable()
export class UsersService {
    constructor(@InjectModel(User.name) private userModel: Model<UserDocument>) {}

    async findByEmail(email: string): Promise<User | null> {
        return this.userModel.findOne({email}).exec();
    }

    async createUser(email: string, password: string, name: string): Promise<User> {
        const newUser = new this.userModel({ email, password, name });
        return newUser.save();
    }
}