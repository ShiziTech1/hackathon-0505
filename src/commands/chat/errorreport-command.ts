import { ActionRowBuilder, ButtonBuilder, ButtonStyle, ChatInputCommandInteraction, PermissionsString } from 'discord.js';
import { RateLimiter } from 'discord.js-rate-limiter';

import { Language } from '../../models/enum-helpers/index.js';
import { EventData } from '../../models/internal-models.js';
import { Lang } from '../../services/index.js';
import { InteractionUtils } from '../../utils/index.js';
import { Command, CommandDeferType } from '../index.js';
import axios from 'axios';

export class ErrorReportCommand implements Command {
    public names = [Lang.getRef('chatCommands.errorreport', Language.Default)];
    public cooldown = new RateLimiter(1, 5000);
    public deferType = CommandDeferType.HIDDEN;
    public requireClientPerms: PermissionsString[] = [];

    public async execute(intr: ChatInputCommandInteraction, data: EventData): Promise<void> {
        const errorLog = intr.options.getString("input");
        let response = await axios.post(`http://localhost:8000/ask?question="Here is the source code and the log. Please provide: 1.) The line where the error occurs, 2.) The reason for the error, and 3.) How to fix it."`);
        // const row = new ActionRowBuilder<ButtonBuilder>()
        //     .addComponents(
        //         new ButtonBuilder()
        //             .setCustomId('button1')
        //             .setLabel('Primary')
        //             .setStyle(ButtonStyle.Primary),
        //         new ButtonBuilder()
        //             .setCustomId('button2')
        //             .setLabel('Secondary')
        //             .setStyle(ButtonStyle.Secondary),
        //         new ButtonBuilder()
        //             .setCustomId('button3')
        //             .setLabel('Success')
        //             .setStyle(ButtonStyle.Success)
        //     );
        await InteractionUtils.send(intr, response.data.result);
    }
}
