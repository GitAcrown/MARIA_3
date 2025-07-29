import io
import asyncio
import logging
import numexpr as ne
from pathlib import Path
from typing import Literal, Union

import discord
from discord import Interaction, app_commands
from discord.ext import commands
from openai import AsyncOpenAI

from common import dataio
from common.llm.agents import *
from common.llm.classes import *

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTES ----------------------------------------------------

TRANSCRIPTION_MODEL = 'whisper-1'
MATH_ANSWER_MODEL = 'gpt-4.1-nano'

PROPOSAL_EMOJI = '<:suggestion:1399517830394937664>'
PROPOSAL_TYPES = Literal['audio_transcription']

# CLASSES ----------------------------------------------------
    
class AudioTranscription:
    def __init__(self, 
                 cog: 'Auto',
                 client: AsyncOpenAI):
        self.cog = cog
        self.client = client
    
    # Transcription audio
    async def fetch_message_audio(self, message: discord.Message) -> io.BytesIO | Path | None:
        """Extrait le texte d'un message audio (vocal discord seulement)."""
        for attachment in message.attachments:
            # Message audio Discord
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
        return None
    
    async def get_audio_transcript(self,
                                        audio_file: io.BytesIO | Path | str,
                                        *,
                                        close_binary: bool = True,
                                        unlink_path: bool = True) -> str:
        if isinstance(audio_file, io.BytesIO):
            audio_file.seek(0)
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file
            )
        except openai.BadRequestError as e:
            logger.error(f"Erreur de transcription : {e}")
            raise e
        except openai.OpenAIError as e:
            logger.error(f"Erreur OpenAI : {e}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inconnue : {e}")
            raise e
        
        if isinstance(audio_file, io.BytesIO) and close_binary:
            audio_file.close()
        if isinstance(audio_file, (str, Path)) and unlink_path:
            try:
                os.unlink(audio_file)
            except OSError:
                pass
        return transcript.text
    
    async def get_transcription(self, message: discord.Message) -> str | None:
        """Récupère la transcription d'un message audio."""
        audio_file = await self.fetch_message_audio(message)
        if not audio_file:
            return None
        
        try:
            transcript = await self.get_audio_transcript(audio_file)
            return transcript.strip()
        except Exception as e:
            logger.error(f"Erreur lors de la transcription audio : {e}")
            return None

# COG ----------------------------------------------------
class Auto(commands.Cog):
    """Cog pour les fonctionnalités de suggestions automatiques de l'IA."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        guild_settings = dataio.DictTableBuilder(
            name='guild_settings',
            default_values={
                'proposal_expiration': 180,
                'suggest_audio_transcription': True
            }
        )
        self.data.map_builders(discord.Guild, guild_settings)
        
        self._gptclient = AsyncOpenAI(
            api_key=self.bot.config['OPENAI_API_KEY']
        )
        
        self._transcription_agent = AudioTranscription(
            cog=self,
            client=self._gptclient
        )
        self._proposals : dict[int, set[PROPOSAL_TYPES]] = {}
            
    # CONFIGURATION MANAGEMENT ------------------------------------
    
    def get_guild_config(self, guild: discord.Guild, key: str):
        """Récupère la configuration d'une guilde."""
        return self.data.get(guild).get_dict_value('guild_settings', key)
    
    def set_guild_config(self, guild: discord.Guild, key: str, value: Union[str, int, bool]) -> None:
        """Met à jour la configuration d'une guilde."""
        self.data.get(guild).set_dict_value('guild_settings', key, value)
            
    # PROPOSALS MANAGEMENT ------------------------------------
    
    def add_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> None:
        """Ajoute une proposition de service automatique."""
        if message.id not in self._proposals:
            self._proposals[message.id] = set()
        self._proposals[message.id].add(proposal_type)
            
    def remove_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> None:
        """Supprime une proposition de service automatique."""
        if message.id in self._proposals and proposal_type in self._proposals[message.id]:
            self._proposals[message.id].remove(proposal_type)
            if not self._proposals[message.id]:
                del self._proposals[message.id]
                
    def has_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> bool:
        """Vérifie si une proposition de service automatique existe pour le message."""
        return message.id in self._proposals and proposal_type in self._proposals[message.id]
    
    def get_proposals(self, message: discord.Message) -> set[PROPOSAL_TYPES]:
        """Récupère toutes les propositions pour un message donné."""
        return self._proposals.get(message.id, set())
        
    async def _schedule_proposals_expiration(self, message: discord.Message, expiration: int) -> None:
        """Programme l'expiration des propositions pour un message."""
        await asyncio.sleep(expiration)
        if message.id in self._proposals:
            # Supprime toutes les propositions après expiration
            try:
                await message.clear_reaction(PROPOSAL_EMOJI)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                pass
            self._proposals.pop(message.id, None)
        
    # LISTENER ----------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Écoute les messages pour déclencher des actions automatiques."""
        if message.author.bot or not message.guild:
            return
        
        # Ignore les messages dans les salons privés
        if isinstance(message.channel, discord.DMChannel):
            return
        
        any_proposal = False
        expiration = int(self.get_guild_config(message.guild, 'proposal_expiration'))
        
        attachments = message.attachments
        if attachments:
            if any(attachment.content_type and attachment.content_type.startswith('audio') for attachment in attachments):
                # Si le message contient un fichier audio, on transcrit
                if bool(self.get_guild_config(message.guild, 'suggest_audio_transcription')):
                    expiration = int(self.get_guild_config(message.guild, 'proposal_expiration'))
                    self.add_proposal(message, 'audio_transcription')
                    try:
                        await message.add_reaction(PROPOSAL_EMOJI)
                    except (discord.Forbidden, discord.HTTPException) as e:
                        logger.warning(f"Impossible d'ajouter une réaction: {e}")
                        self.remove_proposal(message, 'audio_transcription')
        else:
            # Si le message n'a pas d'attachement, on vérifie les propositions
            if self.has_proposal(message, 'audio_transcription'):
                # Si une proposition audio transcription existe, on la supprime
                try:
                    await message.clear_reaction(PROPOSAL_EMOJI)
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    pass  # Ignore les erreurs de permissions
                self.remove_proposal(message, 'audio_transcription')
        
        if any_proposal:
            # Si une proposition a été ajoutée, on programme son expiration
            if expiration > 0:
                asyncio.create_task(self._schedule_proposals_expiration(message, expiration))
                    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Écoute les réactions pour traiter les propositions."""
        if user.bot:
            return
        
        if str(reaction.emoji) != PROPOSAL_EMOJI:
            return
        
        message = reaction.message
        
        if self.has_proposal(message, 'audio_transcription'):
            try:
                async with message.channel.typing():
                    # Traite la transcription audio
                    transcription = await self._transcription_agent.get_transcription(message)
                    if transcription:
                        content = f">>> {transcription}\n-# Transcription demandée par {user.mention}"
                        await message.reply(content, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                    else:
                        await message.reply("Aucune transcription disponible pour ce message audio.", mention_author=False, allowed_mentions=discord.AllowedMentions.none(), delete_after=10)
                    
            except Exception as e:
                logger.error(f"Erreur lors de la transcription audio: {e}")
                await message.reply("Erreur lors de la transcription audio.", mention_author=False, allowed_mentions=discord.AllowedMentions.none())
            
            finally:
                # Supprime la proposition et la réaction à la fin
                try:
                    await message.clear_reaction(PROPOSAL_EMOJI)
                except discord.Forbidden:
                    # Le bot n'a pas les permissions pour supprimer les réactions
                    logger.warning(f"Permissions insuffisantes pour supprimer les réactions dans {message.guild.name}")
                except discord.NotFound:
                    # Le message ou la réaction n'existe plus
                    pass
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression de la réaction: {e}")
                
                self.remove_proposal(message, 'audio_transcription')
                
    # COMMANDS ----------------------------------------------------
    
    auto_group = app_commands.Group(name='auto', description="Paramètres des fonctionnalités automatiques de l'IA", default_permissions=discord.Permissions(manage_messages=True))
    
    @auto_group.command(name='proposal_expiration')
    async def proposal_expiration(self, interaction: Interaction, duration: app_commands.Range[int, 30, 600]):
        """Modifie la durée de validité des propositions
        
        :param duration: Durée en secondes (30 à 600 secondes)"""
        self.set_guild_config(interaction.guild, 'proposal_expiration', duration)
        await interaction.response.send_message(f"**Durée de validité des propositions modifiée** ⸱ `{duration} secondes`", ephemeral=True)
    
    @auto_group.command(name='transcription')
    async def transcription(self, interaction: Interaction, status: bool):
        """Active ou désactive la suggestion de transcription audio
        
        :param status: `True` pour activer, `False` pour désactiver"""
        self.set_guild_config(interaction.guild, 'suggest_audio_transcription', status)
        if status:
            await interaction.response.send_message("**Suggestions de transcription audio activées**", ephemeral=True)
        else:
            await interaction.response.send_message("**Suggestions de transcription audio désactivées**", ephemeral=True)

async def setup(bot):
    await bot.add_cog(Auto(bot))
