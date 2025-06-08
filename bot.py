import asyncio
import logging
import os
import sys
import shutil
from typing import Literal, Optional
from pathlib import Path

import discord
import subprocess
from discord import app_commands
from discord.ext import commands
from dotenv import dotenv_values

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s (%(name)s %(module)s) %(message)s",
)
logger = logging.getLogger('MARIA3.Main')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

def cleanup_temp():
    """Nettoie le dossier temp au démarrage."""
    temp_dir = Path('./temp')
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info("Dossier temp nettoyé")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du dossier temp : {e}")
    temp_dir.mkdir(exist_ok=True)

async def load_cogs(bot):
    for folder in os.listdir("./cogs/"):
        try:
            ext = folder
            await bot.load_extension(f"cogs.{ext}.{ext}")
            logger.info(f"Loaded cog: {ext}")
        except Exception as e:
            logger.error(f"Error loading cog {ext}: {type(e).__name__}: {e}")

async def main():
    # Nettoyage du dossier temp au démarrage
    cleanup_temp()
    
    bot = commands.Bot(
        command_prefix="m!",
        intents=intents,
        help_command=None,
        version='3.0 (Preview)'
    )
    bot.config = dotenv_values('.env') # type: ignore
    
    if "TOKEN" not in bot.config: # type: ignore
        logger.error("Missing TOKEN in .env")
        return

    async with bot:
        logger.info("Loading cogs...")
        await load_cogs(bot)
        logger.info("Cogs loaded.")
        
        @bot.command(name='shutdown')
        @commands.is_owner()
        async def shutdown(ctx: commands.Context):
            await ctx.send("Arrêt du bot...")
            await bot.close()

        @bot.command(name='restart')
        @commands.is_owner()
        async def restart(ctx: commands.Context):
            await ctx.send("Redémarrage du bot...")
            await bot.close()
            os.execv(sys.executable, ['python'] + sys.argv)

        @bot.command(name='update')
        @commands.is_owner()
        async def update(ctx: commands.Context):
            await ctx.send("Mise à jour et redémarrage du bot...")
            try:
                result = subprocess.run(['git', 'pull'], capture_output=True, text=True)
                if result.returncode == 0:
                    await ctx.send(f"```\n{result.stdout}\n```")
                    await bot.close()
                    os.execv(sys.executable, ['python'] + sys.argv)
                else:
                    await ctx.send(f"Erreur lors de la mise à jour :\n```\n{result.stderr}\n```")
            except Exception as e:
                await ctx.send(f"Erreur lors de la mise à jour : {str(e)}")

        @bot.event
        async def on_ready():
            print(f"Connecté en tant que {bot.user}")
            print(f"version discord.py : {discord.__version__}")
            print("> Invitation (ADMIN) : {}".format(discord.utils.oauth_url(int(bot.config["APP_ID"]), permissions=discord.Permissions(8)))) # type: ignore
            print(f"Connecté à {len(bot.guilds)} serveurs :\n" + '\n'.join([f"- {guild.name} ({guild.id})" for guild in bot.guilds]))
            print("--------------")
    
        @bot.tree.error
        async def on_command_error(interaction: discord.Interaction, error):
            if isinstance(error, app_commands.errors.CommandOnCooldown):
                minutes, seconds = divmod(error.retry_after, 60)
                hours, minutes = divmod(minutes, 60)
                hours = hours % 24
                msg = f"**Cooldown ·** Tu pourras réutiliser la commande dans {f'{round(hours)} heures' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} secondes' if round(seconds) > 0 else ''}."
                return await interaction.response.send_message(content=msg, ephemeral=True)
            elif isinstance(error, app_commands.errors.MissingPermissions):
                msg = f"**Erreur ·** Tu manques des permissions `" + ", ".join(error.missing_permissions) + "` pour cette commande !"
                return await interaction.response.send_message(content=msg)
            else:
                logger.error(f'Erreur App_commands : {error}', exc_info=True)
                if interaction.response:
                    if interaction.response.is_done():
                        try:
                            await interaction.followup.send(content=f"**Erreur ·** Une erreur est survenue lors de l'exécution de la commande :\n`{error}`")
                        except discord.HTTPException:
                            pass
                    return await interaction.response.send_message(content=f"**Erreur ·** Une erreur est survenue lors de l'exécution de la commande :\n`{error}`", delete_after=45)
        
        # Synchronisation des commandes ---------------------------
        
        @bot.command(name='sync')
        @commands.guild_only()
        @commands.is_owner()
        async def sync(ctx: commands.Context, guilds: commands.Greedy[discord.Object], spec: Optional[Literal["~", "*", "^"]] = None) -> None:
            """Synchronisation des commandes localement ou globalement
            
            sync -> Synchronise toutes les commandes globales
            sync ~ -> Synchronise le serveur actuel
            sync * -> Copie les commandes globales vers le serveur actuel et synchronise
            sync ^ -> Supprime toutes les commandes du serveur actuel et synchronise
            sync id_1 id_2 -> Synchronise les serveurs id_1 et id_2
            """
            if not guilds:
                if spec == "~":
                    synced = await ctx.bot.tree.sync(guild=ctx.guild)
                elif spec == "*":
                    ctx.bot.tree.copy_global_to(guild=ctx.guild)
                    synced = await ctx.bot.tree.sync(guild=ctx.guild)
                elif spec == "^":
                    ctx.bot.tree.clear_commands(guild=ctx.guild)
                    await ctx.bot.tree.sync(guild=ctx.guild)
                    synced = []
                else:
                    synced = await ctx.bot.tree.sync()

                await ctx.send(
                    f"Synchronisation de {len(synced)} commandes {'globales' if spec is None else 'au serveur actuel'} effectuée : {', '.join((f'`{c.name}`' for c in synced))}."
                )
                return

            ret = 0
            for guild in guilds:
                try:
                    await ctx.bot.tree.sync(guild=guild)
                except discord.HTTPException:
                    pass
                else:
                    ret += 1

            await ctx.send(f"Arbre synchronisé dans {ret}/{len(guilds)}.")
            
        await bot.start(bot.config['TOKEN']) # type: ignore
            
if __name__ == "__main__":
    asyncio.run(main())
