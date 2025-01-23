from rapidfuzz import process, fuzz
from src.utils.exceptions import BookNotFoundError, AmbiguousBookError, InvalidParameterError
from src.utils.config import BookRecommenderConfig as cfg

class BookFinder:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        
    def find_book_name(self, input_title, input_author=None):
        """
        Find a book by title and optionally author.
        
        Args:
            input_title (str): The title of the book
            input_author (str, optional): The author of the book
            
        Returns:
            tuple: (title, author) if found
            
        Raises:
            BookNotFoundError: If the book cannot be found
            AmbiguousBookError: If multiple books match the query
            InvalidParameterError: If input parameters are invalid
        """
        if not isinstance(input_title, str) or (input_author is not None and not isinstance(input_author, str)):
            raise InvalidParameterError("Title and author must be strings")
        
        if not input_title.strip():
            raise InvalidParameterError("Title cannot be empty")
            
        input_title_lower = input_title.lower()
        
        try:
            if input_author:
                # If author is provided, try exact match with title and author
                input_author_lower = input_author.lower()
                if (input_title_lower, input_author_lower) in self.data_processor.title_author_lookup:
                    return self.data_processor.title_author_lookup[(input_title_lower, input_author_lower)]
                
                # If no exact match, try fuzzy matching with both title and author
                matches = process.extract(
                    input_title,
                    [(t, a) for t, a in self.data_processor.title_author_pairs 
                     if a.lower() == input_author_lower],
                    scorer=fuzz.ratio,
                    limit=1,
                    score_cutoff=cfg.FUZZY_MATCH_THRESHOLD
                )
                if matches:
                    return matches[0][0]
                raise BookNotFoundError(
                    f"No book found with title '{input_title}' by author '{input_author}'"
                )
            else:
                # If no author provided, try exact match with title only
                matching_pairs = [
                    (t, a) for (t_lower, a_lower), (t, a) 
                    in self.data_processor.title_author_lookup.items() 
                    if t_lower == input_title_lower
                ]
                if len(matching_pairs) == 1:
                    return matching_pairs[0]
                elif len(matching_pairs) > 1:
                    authors = [pair[1] for pair in matching_pairs]
                    raise AmbiguousBookError(
                        f"Multiple books found with title '{input_title}' by authors: {authors}. "
                        "Please specify an author to get accurate recommendations."
                    )
                
                # If no exact match, try fuzzy matching with title only
                match = process.extractOne(
                    input_title, 
                    self.data_processor.book_titles,
                    scorer=fuzz.ratio,
                    score_cutoff=cfg.FUZZY_MATCH_THRESHOLD
                )
                if match:
                    title = match[0]
                    matching_pairs = [
                        (t, a) for t, a in self.data_processor.title_author_pairs 
                        if t == title
                    ]
                    if len(matching_pairs) == 1:
                        return matching_pairs[0]
                    else:
                        authors = [pair[1] for pair in matching_pairs]
                        raise AmbiguousBookError(
                            f"Multiple books found with similar title '{title}' by authors: {authors}. "
                            "Please specify an author to get accurate recommendations."
                        )
                
                raise BookNotFoundError(f"No book found with title '{input_title}'")
        except Exception as e:
            if not isinstance(e, (BookNotFoundError, AmbiguousBookError, InvalidParameterError)):
                raise BookNotFoundError(f"An error occurred while finding the book: {str(e)}")
            raise 