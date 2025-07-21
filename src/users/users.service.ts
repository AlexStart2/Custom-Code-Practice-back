import { Injectable, HttpException } from '@nestjs/common';
import { Model } from 'mongoose';
import { User, UserDocument, History, HistoryDocument } from './schemas/user.schema';
import { InjectModel } from '@nestjs/mongoose';
import { Types } from 'mongoose';


@Injectable()
export class UsersService {
    constructor(@InjectModel(User.name) private userModel: Model<UserDocument>,
                @InjectModel(History.name) private historyModel: Model<HistoryDocument>) {}

    async findByEmail(email: string): Promise<User | null> {
        return this.userModel.findOne({email}).exec();
    }

    async createUser(email: string, password: string, name: string): Promise<User> {
        const newUser = new this.userModel({ email, password, name });
        new this.historyModel({ 
            owner: newUser._id, 
            dataset: null, 
            messages: [], 
            model: '', 
            createdAt: new Date() 
        }).save();
        return newUser.save();
    }

    async getRagQueryHistory(userId: string): Promise<History | null> {

        if (!Types.ObjectId.isValid(userId)) {
            throw new HttpException('Invalid user ID format', 400);
        }

        const user = await this.userModel.findById(userId).exec();
        if (!user) {
            throw new HttpException('User not found', 404);
        }

        const history = await this.historyModel.find({ owner: user._id }).exec();

        if (!history) {
            throw new HttpException('History not found', 404);
        }

        return history[0];
    }
}