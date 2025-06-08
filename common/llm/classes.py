"""### LLM > Classes
Contient les classes utilisées par les agents."""

import json
import re
import inspect
from datetime import datetime
from typing import Callable, Iterable, Literal, Any, Union, Awaitable

import discord
import tiktoken
import unidecode
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

# CONSTANTS ------------------------------------------------------------------
GPT_TOKENIZER = tiktoken.get_encoding('cl100k_base')

# UTILS ----------------------------------------------------------------------
def _sanitize_text(text: str) -> str:
    """Retire les caractères spéciaux d'un texte."""
    text = ''.join([c for c in unidecode.unidecode(text) if c.isalnum() or c.isspace()]).rstrip()
    return re.sub(r"[^a-zA-Z0-9_-]", "", text[:32])

# MESSAGE COMPONENTS ----------------------------------------------------------

class MessageContentComponent:
    def __init__(self, **data):
        self.data = data
        self.token_count = 0

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)
    
    @property
    def type(self):
        return self.data.get('type', 'text')
    
class TextComponent(MessageContentComponent):
    def __init__(self, text: str, **kwargs: Any):
        super().__init__(type='text', text=text, **kwargs)
        self.token_count = len(GPT_TOKENIZER.encode(text))

    @property
    def payload(self) -> dict:
        return {
            'type': self.type,
            'text': self.data['text']
        }
    
class JSONComponent(MessageContentComponent):
    def __init__(self, json_data: dict, **kwargs: Any):
        super().__init__(type='text', text=json.dumps(json_data), **kwargs)
        self.token_count = len(GPT_TOKENIZER.encode(self.data['text']))
    
    @property
    def payload(self) -> dict:
        return {
            'type': self.type,
            'text': self.data['text']
        }
    
class MetadataTextComponent(MessageContentComponent):
    def __init__(self, _data_title: str, **data):
        _text = f'<{_data_title.upper()} ' + ' '.join([f'{k.lower()}={v}' for k, v in data.items()]) + '>' if data else f'<{_data_title.upper()}>'
        super().__init__(type='text', text=_text, **data)
        self.token_count = len(GPT_TOKENIZER.encode(_text))
    
    @property
    def payload(self) -> dict:
        return {
            'type': self.type,
            'text': self.data['text']
        }
        
class ImageURLComponent(MessageContentComponent):
    def __init__(self, url_path: str, detail: Literal['low', 'high', 'auto'] = 'auto', **kwargs: Any):
        super().__init__(type='image_url', image_url={'url': url_path, 'detail': detail}, **kwargs)
        self.token_count = 250 # Estimation approx.

    @property
    def payload(self) -> dict:
        return {
            'type': self.type,
            'image_url': {
                'url': self.data['image_url']['url'],
                'detail': self.data['image_url']['detail']
            }
        }
        
# ATTACHMENTS -----------------------------------------------------------------

class MessageAttachment:
    def __init__(self, attachment: discord.Attachment, **kwargs: Any):
        self.attachment = attachment
        self.kwargs = kwargs

class AudioAttachment(MessageAttachment):
    def __init__(self, attachment: discord.Attachment, **kwargs: Any):
        super().__init__(attachment, **kwargs)
        self.type = 'audio'

class VideoAttachment(MessageAttachment):
    def __init__(self, attachment: discord.Attachment, **kwargs: Any):
        super().__init__(attachment, **kwargs)
        self.type = 'video'

class TextFileAttachment(MessageAttachment):
    def __init__(self, attachment: discord.Attachment, **kwargs: Any):
        super().__init__(attachment, **kwargs)
        self.type = 'text_file'

# MESSAGE ---------------------------------------------------------------------

class ContextMessage:
    def __init__(self,
                 role: Literal['user', 'assistant', 'developer', 'tool'],
                 components: Iterable[MessageContentComponent] | MessageContentComponent,
                 *,
                 name: str | None = None,
                 attachments: Iterable[MessageAttachment] | MessageAttachment = [],
                 **kwargs: Any):
        self.role = role
        self.components = [components] if isinstance(components, MessageContentComponent) else list(components)
        self.name = _sanitize_text(name) if name else None
        self.attachments = [attachments] if isinstance(attachments, MessageAttachment) else list(attachments)
        self.kwargs = kwargs

        self._created_at = datetime.now()

    def __repr__(self) -> str:
        return f"<ContextMessage type={self.__class__.__name__} role={self.role} components={self.components} attachments={self.attachments}>"

    @property
    def token_count(self):
        return sum(component.token_count for component in self.components)
        
    # Contenu --------------------------------
    def add_components(self, *components: MessageContentComponent) -> None:
        if not isinstance(components, MessageContentComponent):
            components = list(components)
        self.components.extend(components)
        
    def remove_components(self, *components: MessageContentComponent) -> None:
        for component in components:
            if isinstance(component, MessageContentComponent):
                self.components.remove(component)
            else:
                raise ValueError(f"Le composant {component} n'est pas un composant valide.")
            
    def clear_components(self) -> None:
        self.components = []
        
    def get_components(self, filter: Callable[[MessageContentComponent], bool] | None = None) -> list[MessageContentComponent]:
        if filter is None:
            return self.components
        return [component for component in self.components if filter(component)]
    
    # Attachements --------------------------------
    def add_attachments(self, *attachments: MessageAttachment) -> None:
        if not isinstance(attachments, MessageAttachment):
            attachments = list(attachments)
        self.attachments.extend(attachments)
        
    def remove_attachments(self, *attachments: MessageAttachment) -> None:
        for attachment in attachments:
            if isinstance(attachment, MessageAttachment):
                self.attachments.remove(attachment)
            else:
                raise ValueError(f"L'attachement {attachment} n'est pas un attachement valide.")
            
    def clear_attachments(self) -> None:
        self.attachments = []
        
    def get_attachments(self, filter: Callable[[MessageAttachment], bool] | None = None) -> list[MessageAttachment]:
        if filter is None:
            return self.attachments
        return [attachment for attachment in self.attachments if filter(attachment)]
        
    # Helpers --------------------------------
    @property
    def contains_text(self) -> bool:
        return any(component.type == 'text' for component in self.components)
    
    @property
    def contains_image(self) -> bool:
        return any(component.type == 'image_url' for component in self.components)
    
    @property
    def attachments_processed(self) -> bool:
        return not self.attachments
    
    @property
    def full_text(self) -> str:
        return ''.join(component.data['text'] for component in self.components if component.type == 'text')
    
    # Communication avec l'API ---------------
    @property
    def payload(self) -> dict:
        payload = {
            'role': self.role,
            'content': [c.payload for c in self.components]
        }
        if self.name:
            payload['name'] = self.name
        return payload
    
class DeveloperMessage(ContextMessage):
    def __init__(self, prompt: str, **kwargs: Any):
        super().__init__(role='developer', components=TextComponent(prompt), **kwargs)

    def __repr__(self) -> str:
        return f"<DeveloperMessage length={self.components[0].token_count}>"

class AssistantMessage(ContextMessage):
    def __init__(self,
                 components: Iterable[MessageContentComponent] | MessageContentComponent,
                 **kwargs: Any):
        super().__init__(role='assistant', components=components, **kwargs)

    def __repr__(self) -> str:
        return f"<AssistantMessage components={self.components} tool_calls={self.tool_calls}>"

    @property
    def finish_reason(self) -> str | None:
        return self.kwargs.get('finish_reason', None)
    
    @property
    def tool_calls(self) -> list['ToolCall']:
        return [ToolCall.from_message_tool_call(tool_call) for tool_call in self.kwargs.get('tool_calls', [])]
    
    @property
    def is_empty(self) -> bool:
        return not self.components
    
    @classmethod
    def from_chat_completion(cls, chat_completion: ChatCompletion) -> 'AssistantMessage':
        if not chat_completion.choices:
            raise ValueError("ChatCompletion has no choices")
        choice = chat_completion.choices[0]
        usage = chat_completion.usage.completion_tokens if chat_completion.usage else 0
        tool_calls = choice.message.tool_calls if choice.message and choice.message.tool_calls else []
        return cls(
            components=TextComponent(choice.message.content) if choice.message and choice.message.content else MetadataTextComponent('EMPTY'),
            token_count=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )
    
    @property
    def payload(self) -> dict:
        if self.tool_calls:
            return {
                'role': self.role,
                'tool_calls': [t.data for t in self.tool_calls]
            }
        return super().payload
    
class UserMessage(ContextMessage):
    def __init__(self,
                 components: Iterable[MessageContentComponent] | MessageContentComponent,
                 name: str = 'user',
                 **kwargs: Any):
        super().__init__(role='user', components=components, name=name, **kwargs)

        self.message : discord.Message = kwargs.get('message', None)
    
    def __repr__(self) -> str:
        return f"<UserMessage components={self.components} attachments={self.attachments}>"

    @classmethod
    def from_discord_message(cls, message: discord.Message,
                             context_format: str = '[{message.id}] {message.author.name} ({message.author.id})',
                             include_embeds: bool = True,
                             include_attachments: bool = True,
                             include_stickers: bool = True) -> 'UserMessage':
        components = []
        attachments = []
            
        if message.content:
            user_form = context_format.format(message=message, author=message.author)
            components.append(TextComponent(f"{user_form}: {message.clean_content}"))
            
            # Images en URL
            for match in re.finditer(r'(https?://[^\s]+)', message.content):
                url = match.group(0)
                cleanurl = re.sub(r'\?.*$', '', url)
                if cleanurl.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    components.append(ImageURLComponent(cleanurl, detail='auto'))
            
        # Embeds
        if include_embeds:
            for embed in message.embeds:
                components.append(MetadataTextComponent('EMBED', title=embed.title, description=embed.description, url=embed.url))
                if embed.image and embed.image.url:
                    components.append(ImageURLComponent(embed.image.url, detail='high'))
                if embed.thumbnail and embed.thumbnail.url:
                    components.append(ImageURLComponent(embed.thumbnail.url, detail='low'))
                
        # Attachments
        if include_attachments:
            for attachment in message.attachments:
                if attachment.content_type:
                    if attachment.content_type.startswith('image/'):
                        components.append(ImageURLComponent(attachment.url, detail='auto'))
                    elif attachment.content_type.startswith('audio/'):
                        attachments.append(AudioAttachment(attachment))
                    elif attachment.content_type.startswith('video/'):
                        attachments.append(VideoAttachment(attachment))
                    # elif attachment.content_type.startswith('text/'):
                        # attachments.append(TextFileAttachment(attachment))
            
        # Stickers
        if include_stickers:
            for sticker in message.stickers:
                if sticker.url:
                    components.append(ImageURLComponent(sticker.url, detail='auto'))
                
        return cls(
            components=components,
            name=message.author.name,
            attachments=attachments,
            message=message
        )
    
class ToolResponseMessage(ContextMessage):
    def __init__(self,
                 data: dict,
                 tool_call_id: str,
                 **kwargs: Any):
        super().__init__(role='tool', components=TextComponent(json.dumps(data)), **kwargs)
        self.response_data = data
        self.tool_call_id = tool_call_id

    def __repr__(self) -> str:
        return f"<ToolResponseMessage response_data={self.response_data} tool_call_id={self.tool_call_id}>"
        
    @property
    def is_empty(self) -> bool:
        return not self.components
    
    @property
    def is_error(self) -> bool:
        return self.response_data.get('error', False)
        
    @property
    def payload(self) -> dict:
        return {
            'role': self.role,
            'content': [c.payload for c in self.components],
            'tool_call_id': self.tool_call_id
        }
    
    # Helpers --------------------------------
    @property
    def header(self) -> str:
        return self.kwargs.get('header', '')
    
# TOOLS ----------------------------------------------------------------------

class ToolCall:
    def __init__(self, **data):
        self.data = data
        
    def __repr__(self) -> str:
        return f"<ToolCall {self.data}>"
    
    @property
    def function_name(self) -> str:
        return self.data['function']['name']
    
    @property
    def arguments(self) -> dict:
        return json.loads(self.data['function']['arguments'])
    
    @classmethod
    def from_message_tool_call(cls, message_tool_call: ChatCompletionMessageToolCall) -> 'ToolCall':
        return cls(
            id=message_tool_call.id,
            type='function',
            function={
                'name': message_tool_call.function.name,
                'arguments': message_tool_call.function.arguments
            }
        )
        
class Tool:
    def __init__(self,
                 name: str,
                 description: str,
                 properties: dict,
                 function: Union[Callable[[ToolCall, 'MessageGroup'], ToolResponseMessage], 
                               Callable[[ToolCall, 'MessageGroup'], Awaitable[ToolResponseMessage]]],
                 **extras: Any):
        self.name = name
        self.description = description
        self.properties = properties
        self.function = function
        self.extras = extras
        
        self._required = [k for k, _ in properties.items()]
    
    async def execute(self, tool_call: ToolCall, context: 'MessageGroup') -> ToolResponseMessage:
        """Exécute la fonction de l'outil. Supporte les fonctions synchrones et asynchrones."""
        if inspect.iscoroutinefunction(self.function):
            return await self.function(tool_call, context)
        else:
            return self.function(tool_call, context)
    
    @property
    def to_dict(self) -> dict:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'strict': True,
                'parameters': {
                    'type': 'object',
                    'properties': self.properties,
                    'required': self._required,
                    'additionalProperties': False
                }
            }
        }

# GROUPS ----------------------------------------------------------------------
class MessageGroup:
    def __init__(self,
                 messages: Iterable[ContextMessage],
                 **kwargs: Any):
        self.messages = list(messages)
        self.kwargs = kwargs
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MessageGroup):
            return NotImplemented
        return self.messages == other.messages
    
    @property
    def total_token_count(self) -> int:
        return sum([m.token_count for m in self.messages])
    
    # Gestion des messages -------------------
    def append_messages(self, *messages: ContextMessage) -> None:
        if not isinstance(messages, ContextMessage):
            messages = list(messages)
        self.messages.extend(messages)
        
    def remove_messages(self, *messages: ContextMessage) -> None:
        if not isinstance(messages, ContextMessage):
            messages = list(messages)
        for m in messages:
            if m in self.messages:
                self.messages.remove(m)
        
    def get_messages(self, filter: Callable[[ContextMessage], bool] = lambda _: True) -> list[ContextMessage]:
        return [m for m in self.messages if filter(m)]
    
    @property
    def last_message(self) -> ContextMessage | None:
        return self.messages[-1] if self.messages else None
    
    @property
    def last_completion(self) -> AssistantMessage | None:
        for m in reversed(self.messages):
            if isinstance(m, AssistantMessage):
                return m
        return None
    
    # Utilitaires ---------------------
    @property
    def received_any_response(self) -> bool:
        return any([isinstance(m, AssistantMessage) for m in self.messages])
    
    @property
    def awaiting_response(self) -> bool:
        return isinstance(self.last_message, UserMessage) or isinstance(self.last_message, ToolResponseMessage)
    
    @property
    def contains_text(self) -> bool:
        return any([m.contains_text for m in self.messages])
    
    @property
    def contains_image(self) -> bool:
        return any([m.contains_image for m in self.messages])
    
    def fetch_author(self) -> discord.User | discord.Member | None:
        for m in self.messages:
            if isinstance(m, UserMessage):
                return m.kwargs.get('message').author
        return None
    
    def fetch_guild(self) -> discord.Guild | None:
        for m in self.messages:
            if isinstance(m, UserMessage):
                return m.kwargs.get('message').guild
        return None
    
    def fetch_channel(self) -> discord.abc.Messageable | None:
        for m in self.messages:
            if isinstance(m, UserMessage):
                return m.kwargs.get('message').channel
        return None

    def search_for_message_components(self, component_type: type[MessageContentComponent], filter: Callable[[MessageContentComponent], bool] | None = None) -> list[MessageContentComponent]:
        return [c for c in self.messages if isinstance(c, component_type) and (filter is None or filter(c))]
